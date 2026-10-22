import unittest
from system import WIDTH, digest, OUT, read, run

class Invariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run()

    def test_all_requirements(self):
        e=read(OUT/'evidence.json')
        self.assertTrue(e['overall_pass'])
        self.assertTrue(all(r['passed'] for r in e['requirements'].values()))

    def test_causal_padding_and_loss(self):
        for b in read(OUT/'ledgers'/'batch_catalog.json'):
            n=sum(b['attention_mask'])
            self.assertEqual(len(b['input_ids']),WIDTH)
            self.assertEqual(sum(b['loss_mask']),n-1)
            self.assertEqual(b['position_ids'][:n],list(range(n)))
            for i in range(WIDTH):
                for j in range(WIDTH):
                    self.assertEqual(b['causal_mask'][i][j],int(i<n and j<n and j<=i))

    def test_exactly_once_and_replay(self):
        entries=read(OUT/'ledgers'/'consumption.json')
        batch=read(OUT/'ledgers'/'batch_catalog.json')
        self.assertEqual([e['batch_hash'] for e in entries],[b['hash'] for b in batch])
        self.assertTrue(read(OUT/'ledgers'/'replay.json')['matched'])

    def test_firewall_and_hashes(self):
        blocked=read(OUT/'ledgers'/'blocked_shards.json')
        self.assertEqual({x['split'] for x in blocked},{'eval','validation'})
        for b in read(OUT/'ledgers'/'batch_catalog.json'):
            m=read(OUT/'manifests'/(b['shard_id']+'.json'))
            self.assertEqual(m['split'],'train')
            self.assertEqual(m['content_hash'],b['content_hash'])

    def test_checkpoint_offsets(self):
        cp=read(OUT/'checkpoints/precrash.json')
        self.assertEqual(cp['next_index'],cp['consumption_offset'])
        self.assertEqual(cp['next_index'],cp['learning_offset'])
        self.assertEqual(cp['state_hash'],digest({k:v for k,v in cp.items() if k!='state_hash'}))

if __name__=='__main__':
    unittest.main()
