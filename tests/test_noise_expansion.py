import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT, config
from noise_expansion import prepare_plan, fingerprint


class ExpansionTests(unittest.TestCase):
    def test_nested_pools_fixed_evaluation_and_source_exclusions(self):
        cfg=config(ROOT/"config.json")
        categories=cfg["classes"]+cfg["seen_noise"]+cfg["unseen_noise"]+[f"extra_{i:02d}" for i in range(37)]
        with tempfile.TemporaryDirectory() as tmp:
            audio=Path(tmp)
            meta=[]
            for fold in range(1,6):
                for cat in categories:
                    name=f"{fold}-{cat}.wav"
                    (audio/name).touch()
                    meta.append(dict(filename=name,fold=str(fold),category=cat,src_file=f"{fold}-{cat}"))
            val=next(r for r in meta if r["fold"]=="4" and r["category"]=="extra_00")
            val["src_file"]="5-extra_01"
            plan=prepare_plan(meta,audio,cfg)
            self.assertEqual(len(plan["cross_split_source_exclusions"]),1)
            self.assertEqual(plan["cross_split_source_exclusions"][0]["removed_split"],"val")
            previous=set()
            target_gains=None
            for stage in plan["stages"]:
                rows=stage["manifest"]["rows"]
                self.assertEqual(fingerprint([r for r in rows if r["split"]=="val"]),plan["val_fingerprint"])
                self.assertEqual(fingerprint([r for r in rows if r["split"]=="test"]),plan["test_fingerprint"])
                paths={r["noise_path"] for r in rows if r["split"]=="train" and r["noise_path"]}
                self.assertTrue(previous<=paths)
                previous=paths
                gains={r["target_path"]:r["target_rms"] for r in rows if r["split"]=="train" and r["target_path"]}
                if target_gains is None: target_gains=gains
                self.assertEqual(gains,target_gains)
                train=[r for r in rows if r["split"]=="train"]
                self.assertTrue(all(r["noise_class"] not in cfg["unseen_noise"] for r in train))


if __name__=="__main__":
    unittest.main()
