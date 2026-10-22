import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_demo as r
class SystemTests(unittest.TestCase):
 def test_tokenizer_frozen(self):
  self.assertEqual(r.tokenize('A'),[65]);self.assertEqual(r.tokenize('తెలుగు'),'తెలుగు'.encode('utf8') and list('తెలుగు'.encode('utf8')))
 def test_hash_canonical(self):self.assertEqual(r.digest({'a':1,'b':2}),r.digest({'b':2,'a':1}))
 def test_curriculum_floors(self):
  for stage in r.STAGES:
   self.assertAlmostEqual(sum(stage['weights'].values()),1)
   for lane,floor in r.FLOORS.items():self.assertGreaterEqual(stage['weights'][lane],floor)
 def test_generated_evidence_and_recovery(self):
  r.main();e=r.read_json(r.OUT/'evidence.json');self.assertTrue(all(v['passed'] for v in e['requirements'].values()))
  a=r.read_json(r.OUT/'ledgers/consumption.json');self.assertEqual([x['offset'] for x in a],list(range(len(a))))
  self.assertEqual(r.read_json(r.OUT/'checkpoints/before_crash.json')['next_offset'],3)
  self.assertEqual(r.read_json(r.OUT/'checkpoints/final.json')['next_offset'],len(a))
  for item in r.read_json(r.OUT/'ledgers/replay.json'):self.assertEqual(item['original_hash'],item['replay_hash'])
 def test_shard_tampering_rejected(self):
  r.main();m=r.read_json(r.OUT/'manifests/train_general.manifest.json');th=r.digest(r.read_json(r.OUT/'manifests/tokenizer.json'))
  self.assertTrue(r.verify_manifest(m,th));p=r.OUT/m['shard_path'];p.write_text('corruption');self.assertFalse(r.verify_manifest(m,th))
if __name__=='__main__':unittest.main()
