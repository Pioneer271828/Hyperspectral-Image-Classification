import itertools
import unittest
from identifiability import support,alias_support


def request(i,mask,selected,context=None):
    return {'context':context or str(i),'view':'v','shape':[4,4],'K':len(mask),'minimum_column':min(mask),
        'maximum_column':max(mask),'endpoint_span_columns':max(mask)-min(mask),'mask_sha256':str(mask),
        'full_selected':{'content_coverage':selected}}


class Identifiability(unittest.TestCase):
    def test_exact_support_matches_exhaustive_enumeration(self):
        rows=[request(i,[0,1,4],i<2) for i in range(4)]
        expected=len(list(itertools.combinations(range(4),2)))
        self.assertEqual(int(support(rows,'legacy')['assignments']),expected)
        self.assertFalse(support(rows,'legacy')['feature_varies_within_randomized_strata']['internal_mask'])

    def test_alias_choices_do_not_create_internal_mask_contrast(self):
        rows=[request(0,[0,1,4],True,'same'),request(1,[0,1,4],False,'same')]
        actual=support(rows,'same_context_count_span')
        self.assertEqual(actual['assignments'],'2');self.assertEqual(actual['alias_quotient_assignments'],'1')
        self.assertFalse(actual['feature_varies_within_randomized_strata']['internal_mask'])

    def test_internal_positive_control_and_span_inertness(self):
        rows=[request(0,[0,1,4],True,'same'),request(1,[0,3,4],False,'same')]
        actual=support(rows,'same_context_count_span')
        self.assertEqual(actual['assignments'],'2');self.assertTrue(actual['feature_varies_within_randomized_strata']['internal_mask'])
        self.assertFalse(actual['feature_varies_within_randomized_strata']['span'])
        self.assertEqual(alias_support([2,2],2),3)


if __name__=='__main__':unittest.main()
