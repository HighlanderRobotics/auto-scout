import unittest
from export_haiku_annotations import sanitize_frame


class ExportAnnotationTests(unittest.TestCase):
    def test_annotation_export_uses_allowlist_and_preserves_robot_evidence(self):
        data={'teacher_model':'claude-haiku-5-5','image_size':[100,100],'usable':True,
            'robots':[{'team':'254','visible':True,'box':[10,10,20,20],'confidence':.8,'private_data':'omit'}],
            'usage':{'input_tokens':123},'api_key':'omit','local_path':'/private/example'}
        row=sanitize_frame(data)
        self.assertNotIn('api_key',row);self.assertNotIn('usage',row)
        self.assertNotIn('local_path',row);self.assertNotIn('private_data',row['robots'][0])
        self.assertEqual(row['robots'][0]['box'],[10,10,20,20])

    def test_invalid_box_is_excluded_by_validator(self):
        with self.assertRaises(ValueError):
            sanitize_frame({'teacher_model':'claude-haiku-5-5','image_size':[100,100],
                'robots':[{'box':[-10,0,20,20],'confidence':.8}]})


if __name__=='__main__':unittest.main()
