import unittest
import sys
from pathlib import Path
import pandas as pd
TEST_DIR=Path(__file__).resolve().parent
CODE_DIR=TEST_DIR.parent/"code"
sys.path.insert(0,str(CODE_DIR if CODE_DIR.is_dir() else TEST_DIR))
from postprocess_communes import parse_number, setup_values, build_panel, MAIN, RAW


def row(year, cid, index, values, *, label='Test', unmapped=False):
    raw=[str(v) if v is not None else '' for v in values]
    return {'row_id':f'row{index}', 'year':str(year),'cassini_id':cid,
      'commune_raw':label,'commune_cleaned':label,'cm_status':'MATCH' if cid else 'NO_MATCH',
      'extraction_flags':'unmapped_columns' if unmapped else '', 'review_flags':'',
      'source_image':f'image-{index//4}',**{f:(str(v) if v is not None and not unmapped else '') for f,v in zip(MAIN,values)},
      **dict(zip(RAW,raw))}


class TestParser(unittest.TestCase):
    def test_thousands_and_francs(self):
        self.assertEqual(parse_number('7,471',False),7471)
        self.assertEqual(parse_number('2,849.27',True),2849.27)
        self.assertEqual(parse_number('2,955,37',True),2955.37)
        self.assertEqual(parse_number('64, 24',True),64.24)
        self.assertEqual(parse_number('1, 181',False),1181)
        self.assertTrue(pd.isna(parse_number('41;10',True)))
    def test_ditto_not_numeric(self):
        self.assertTrue(pd.isna(parse_number('"',False)))


class TestGrouping(unittest.TestCase):
    def test_merges_repeated_core_and_suppresses_conflicts(self):
        vals=['100','200','300','40.20','70']
        dif=['101','200','300','40.20','70']
        data=pd.DataFrame([row(1880,'0001',1,vals),row(1880,'0001',2,vals),
                            row(1880,'0002',3,vals),row(1880,'0002',4,dif),
                            row(1880,'',5,vals)])
        result,validation,discrepancies=setup_values(data,min_anchors=1)
        panel,complete,conflicts,conflict_rows,unmatched=build_panel(result)
        self.assertEqual(len(complete),1)
        self.assertEqual(complete.iloc[0]['cassini_id'],'0001')
        self.assertEqual(complete.iloc[0]['n_candidate_rows'],2)
        self.assertEqual(len(conflicts),1)
        self.assertEqual(conflicts.iloc[0]['cassini_id'],'0002')
        self.assertEqual(len(unmatched),1)

    def test_recovery_requires_layout_confirmation(self):
        val=['100','200','300','40.20','70']
        data=pd.DataFrame([row(1880,'1',1,val), row(1880,'2',2,val,unmapped=True),
            row(1881,'3',3,val,unmapped=True)])
        result,validation,discrepancies=setup_values(data,min_anchors=1)
        self.assertTrue(result.loc[1,'core_complete'])
        self.assertEqual(result.loc[1,'layout_recovered_field_count'],5)
        self.assertFalse(result.loc[2,'core_complete'])
        self.assertEqual(result.loc[2,'layout_recovered_field_count'],0)

if __name__=='__main__':unittest.main()
