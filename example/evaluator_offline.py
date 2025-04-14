import unittest
import torch
import torch.utils.data
from y_rgb.dataset.utils import pseudo_collate, default_collate


class TestCollateFunctions(unittest.TestCase):
    def setUp(self):
        """创建公共测试数据"""
        self.sample_batch = [
            {#[
                'coords': torch.randn(5, 3),
                'charges': torch.tensor([1, 0, 1, 0, 1]),
                'mol_id': 0
            },#]
            {#[
                'coords': torch.randn(5, 3),
                'charges': torch.tensor([0, 1, 0, 1, 0]),
                'mol_id': 1
            }#]
        ]
    # The length of the value of the same key should be consistent
    # for each dictionary/list item

    def test_pseudo_collate_retains_structure(self):
        # this will make mistake
        # because the length of charges is not the same, the first is 4, the second is 5
        hetero_batch = [
            {'coords': torch.randn(5, 4), 'charges': [1, 0, 1, 1]},
            {'coords': torch.randn(5, 3), 'charges': [0, 1, 0, 1, 0]}
        ]

        collated = pseudo_collate(hetero_batch)
        print(collated)

        # 验证字典结构
        self.assertIsInstance(collated, dict)
        # 验证张量未堆叠
        self.assertIsInstance(collated['coords'], list)
        self.assertEqual(len(collated['coords']), 2)
        # 验证列表类型保持
        self.assertIsInstance(collated['charges'][0], list)


    def test_default_collate_tensor_operations(self):
        """验证张量堆叠和设备迁移"""
        collated = default_collate(self.sample_batch)
        # collated = pseudo_collate(self.sample_batch)
        print(collated["coords"])

        # 验证坐标张量维度
        self.assertEqual(collated['coords'].shape, (2, 5, 3))
        # 验证电荷张量类型
        self.assertTrue(torch.is_tensor(collated['charges']))
        # 验证标量转换
        self.assertTrue(torch.is_tensor(collated['mol_id']))
        self.assertEqual(collated['mol_id'].shape, (2,))

    def test_sequence_collation(self):
        """测试序列类型数据处理"""
        seq_batch = [
            (torch.tensor([1, 2]), [3, 4], [5]),
            (torch.tensor([3, 4]), [5, 6], [7])
        ]

        # 测试pseudo模式
        pseudo_result = pseudo_collate(seq_batch)
        print(pseudo_result[1])
        default_result = default_collate(seq_batch)
        print(default_result[1])

        self.assertIsInstance(pseudo_result, list)
        self.assertTrue(torch.equal(pseudo_result[0][0], seq_batch[0][0]))

        # 测试default模式
        default_result = default_collate(seq_batch)
        self.assertIsInstance(default_result, list)
        self.assertEqual(default_result[0].shape, (2, 2))

    def test_device_conflict_detection(self):
        """验证设备冲突检测"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        mixed_device_batch = [
            {'data': torch.tensor([1]).cuda()},
            {'data': torch.tensor([2])}
        ]

        with self.assertRaises(RuntimeError):
            default_collate(mixed_device_batch)

    def test_invalid_batch_error(self):
        """验证尺寸不一致的错误处理"""
        invalid_batch = [
            {'coords': torch.randn(3, 3)},
            {'coords': torch.randn(4, 3)}
        ]

        with self.assertRaises(RuntimeError):
            default_collate(invalid_batch)

    def test_nested_structure_handling(self):
        """测试嵌套数据结构处理"""
        nested_batch = [
            {'features': {'atom': torch.tensor(1), 'bond': [2, 3]}},
            {'features': {'atom': torch.tensor(4), 'bond': [5, 6]}}
        ]

        collated = default_collate(nested_batch)
        print(collated)
        pse_collated = pseudo_collate(nested_batch)
        print(pse_collated)

        # 验证嵌套字典
        self.assertIn('features', collated)
        # 验证张量堆叠
        self.assertEqual(collated['features']['atom'].shape, (2,))
        # 验证列表保持
        self.assertIsInstance(collated['features']['bond'], list)
        self.assertEqual(len(collated['features']['bond'][0]), 2)


if __name__ == '__main__':
    unittest.main()
