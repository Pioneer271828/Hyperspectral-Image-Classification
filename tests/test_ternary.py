"""Independent assignment/gradient/bit-layout oracles and disk recovery."""
import copy
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from compression.ternary import TernaryMatrix, ternarize, assignments
from compression.ternary_codec import encode, decode, save_weights, load_weights

torch.set_num_threads(1)


class TernaryTests(unittest.TestCase):
    def test_threshold_ties_zero_and_single_sign(self):
        for values in ([-2., -1., 0., 1., 2.], [0.]*5, [1.]*5, [-1.]*5):
            w = torch.tensor([values]); p, n = torch.tensor(.7), torch.tensor(.4)
            for ratio in (0., .5, 1.):
                a = np.asarray([values], dtype=np.float32); threshold = abs(a).max()*ratio
                expected = np.where(a > threshold, np.float32(.7), np.where(a < -threshold, np.float32(-.4), np.float32(0)))
                self.assertTrue(np.array_equal(ternarize(w, p, n, ratio).numpy(), expected))
        self.assertEqual(ternarize(torch.tensor([[-2., -1., 1., 2.]]), p, n, .5).tolist(),
                         [[-float(n), 0., 0., float(p)]])

    def test_analytic_scaled_ste_and_signed_scale_derivative(self):
        w = torch.tensor([[-2., -1., 0., 1., 2.]], requires_grad=True)
        p = torch.tensor(.7, requires_grad=True); n = torch.tensor(.4, requires_grad=True)
        upstream = torch.tensor([[2., 3., 5., 7., 11.]])
        (ternarize(w, p, n, .5)*upstream).sum().backward()
        torch.testing.assert_close(w.grad, torch.tensor([[.8, 3., 5., 7., 7.7]]))
        self.assertEqual(float(p.grad), 11.); self.assertEqual(float(n.grad), -2.)
        for which, expected in (('p', 11.), ('n', -2.)):
            eps = .001
            def loss(delta):
                return float((ternarize(w.detach(), p.detach()+delta if which=='p' else p.detach(),
                    n.detach()+delta if which=='n' else n.detach(), .5)*upstream).sum())
            self.assertAlmostEqual((loss(eps)-loss(-eps))/(2*eps), expected, delta=.001)

    def test_forward_input_gradient_bias_and_source_ownership(self):
        torch.manual_seed(2048)
        for source, shape in ((nn.Linear(7, 5), (3, 7)), (nn.Conv2d(7, 5, 1), (2, 7, 3, 4))):
            before = copy.deepcopy(source.state_dict()); layer = TernaryMatrix(source, threshold_ratio=.2)
            x = torch.randn(shape, requires_grad=True); refx = x.detach().clone().requires_grad_()
            w = layer.layer.weight.detach().numpy(); threshold = np.max(abs(w))*.2
            oracle = torch.from_numpy(np.where(w > threshold, float(layer.positive),
                np.where(w < -threshold, -float(layer.negative), 0)).astype(np.float32))
            ref = F.linear(refx, oracle, layer.layer.bias) if source.weight.ndim==2 else F.conv2d(refx, oracle, layer.layer.bias)
            actual = layer(x); self.assertTrue(torch.equal(actual, ref))
            actual.square().sum().backward(); ref.square().sum().backward()
            self.assertTrue(torch.equal(x.grad, refx.grad))
            for k,v in source.state_dict().items(): self.assertTrue(torch.equal(v, before[k]))
            self.assertIsNone(source.weight.grad)

    def test_invalid_layers_dtype_input_and_scale_fail_closed(self):
        for bad in (nn.Conv2d(2, 2, 3), nn.Conv2d(2, 2, 1, groups=2), nn.Linear(2, 2).double()):
            with self.assertRaises(ValueError): TernaryMatrix(bad, threshold_ratio=.1)
        source=nn.Linear(2, 2)
        for ratio in (-1., 1.1, float('nan'), True):
            with self.assertRaises(ValueError): TernaryMatrix(source, threshold_ratio=ratio)
        layer=TernaryMatrix(source, threshold_ratio=.1)
        for x in (torch.ones(2, 2).double(), torch.full((2, 2), float('nan')), torch.ones(2, 3)):
            with self.assertRaises(ValueError): layer(x)
        before=copy.deepcopy(layer.state_dict())
        for bad in (0., -1., float('inf')):
            with torch.no_grad(): layer.positive.fill_(bad)
            with self.assertRaises(ValueError): layer(torch.ones(2, 2))
            self.assertEqual(layer.layer.weight.tolist(), before['layer.weight'].tolist())

    def test_all_zero_master_learns_without_nan_and_missing_sign_gradient(self):
        source=nn.Linear(3, 2, bias=False)
        with torch.no_grad(): source.weight.zero_()
        layer=TernaryMatrix(source, threshold_ratio=.05)
        output=layer(torch.ones(1, 3)); output.sum().backward()
        self.assertEqual(float(output.sum()), 0.)
        self.assertTrue(torch.equal(layer.layer.weight.grad, torch.ones_like(source.weight)))
        self.assertEqual(float(layer.positive.grad), 0.); self.assertEqual(float(layer.negative.grad), 0.)
        opt=torch.optim.SGD(layer.parameters(), lr=.001); opt.step()
        self.assertTrue(torch.isfinite(layer(torch.ones(1, 3))).all())

    def test_actual_two_bit_layout_padding_and_independent_oracle(self):
        for count in range(1, 18):
            values=np.resize(np.array([-2., 0., 3., 1., -1.], dtype=np.float32), count)
            w=torch.from_numpy(values.reshape(1, count)); p,n=torch.tensor(.7),torch.tensor(.4)
            payload=encode(w,p,n,.5); codes=[2 if v < -1.5 else 1 if v > 1.5 else 0 for v in values]
            expected=bytes(sum((codes[j] if j < count else 0) << (2*(j%4)) for j in range(i,min(i+4, count))) for i in range(0,count,4))
            self.assertEqual(payload[21:],expected)
            self.assertEqual(len(payload),21+(count+3)//4)
            target=np.array([0.,float(p),-float(n)],dtype=np.float32)[codes].reshape(1,count)
            self.assertTrue(np.array_equal(decode(payload).numpy(),target))
        w=torch.randn(3, 5, 1, 1); payload=encode(w,p,n,.2)
        self.assertTrue(torch.equal(decode(payload),ternarize(w,p,n,.2)))

    def test_codec_rejects_corrupt_headers_codes_padding_and_lengths(self):
        packet=encode(torch.tensor([[0.,1.,-1.]]),torch.tensor(.5),torch.tensor(.4),.1)
        bad=[b'', b'xxxx'+packet[4:], packet[:4]+b'\x03'+packet[5:], packet[:-1],packet+b'\x00',
             packet[:-1]+bytes([packet[-1]|3]), packet[:-1]+bytes([packet[-1]|0xc0])]
        header=bytearray(packet); struct.pack_into('<f',header,13,-1.); bad.append(bytes(header))
        header=bytearray(packet); struct.pack_into('<I',header,5,0); bad.append(bytes(header))
        for value in bad:
            with self.assertRaises(ValueError): decode(value)

    def test_file_hash_before_decode_no_overwrite_and_size(self):
        torch.manual_seed(64); w=torch.randn(256,512); p,n=torch.tensor(.2),torch.tensor(.3)
        payload=encode(w,p,n,.05)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'weight.tqw'; pin=save_weights(path,payload)
            self.assertEqual(path.stat().st_size,32789)
            self.assertEqual(pin,hashlib.sha256(payload).hexdigest())
            self.assertTrue(torch.equal(load_weights(path,pin),ternarize(w,p,n,.05)))
            with self.assertRaises(FileExistsError): save_weights(path,payload)
            with patch('compression.ternary_codec.decode',side_effect=AssertionError):
                with self.assertRaises(ValueError): load_weights(path,'0'*64)
            self.assertEqual(path.read_bytes(),payload)

    def test_equal_budget_training_disk_optimizer_and_state_recovery(self):
        torch.manual_seed(512); source=nn.Linear(5,3)
        x=torch.randn(16,5); target=torch.randn(16,3); vx=torch.randn(9,5); vy=torch.randn(9,3)
        for name,model in (('fp32',copy.deepcopy(source)),('tq',TernaryMatrix(source,threshold_ratio=.2))):
            initial=copy.deepcopy(model.state_dict()); opt=torch.optim.Adam(model.parameters(),lr=.0001)
            history=[]
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'resume.pt'
                for step in range(12):
                    opt.zero_grad(); F.mse_loss(model(x),target).backward(); opt.step()
                    with torch.no_grad(): history.append(float(F.mse_loss(model(vx),vy)))
                    if step==4: torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),history=history.copy()),path)
                resumed=copy.deepcopy(source) if name=='fp32' else TernaryMatrix(source,threshold_ratio=.2)
                packet=torch.load(path,weights_only=True); resumed.load_state_dict(packet['model'])
                opt2=torch.optim.Adam(resumed.parameters(),lr=.0001); opt2.load_state_dict(packet['optimizer']); replay=packet['history']
                for _ in range(7):
                    opt2.zero_grad(); F.mse_loss(resumed(x),target).backward(); opt2.step()
                    with torch.no_grad(): replay.append(float(F.mse_loss(resumed(vx),vy)))
                for k,v in model.state_dict().items(): self.assertTrue(torch.equal(v,resumed.state_dict()[k]),k)
                self.assertEqual(history,replay)
                self.assertEqual(min(range(12),key=history.__getitem__),min(range(12),key=replay.__getitem__))
                for key in ('positive','negative','layer.weight') if name=='tq' else ('weight',):
                    self.assertFalse(torch.equal(initial[key],model.state_dict()[key]))
                if name=='tq':
                    packed=encode(model.layer.weight,model.positive,model.negative,float(model.threshold_ratio))
                    self.assertTrue(torch.equal(model(vx),F.linear(vx,decode(packed),model.layer.bias)))


if __name__=='__main__': unittest.main()
