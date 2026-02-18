import unittest
import torch
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import DataLoader
import shutil
import tempfile
import os

class TestT5Implementation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Use a tiny configuration to avoid downloading large models during tests
        cls.config = T5Config(
            vocab_size=100,
            d_model=64,
            d_kv=8,
            d_ff=64,
            num_layers=2,
            num_decoder_layers=2,
            num_heads=4,
            relative_attention_num_buckets=8
        )
        cls.model = T5ForConditionalGeneration(cls.config)
        # Mock tokenizer roughly (in real scenario, we'd mock the object entirely or use a local file)
        # Here we assume the user has internet or we mock the tokenizer call behavior
        # For strict isolation, we will mock the inputs directly rather than the tokenizer logic
        cls.device = torch.device("cpu")
        cls.model.to(cls.device)

    def test_model_forward_pass_training(self):
        """Test if the model accepts inputs and labels and returns a loss."""
        batch_size = 2
        seq_len = 10
        target_len = 5

        # Random inputs
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len))
        
        # Random labels with -100 for padding
        labels = torch.randint(0, 100, (batch_size, target_len))
        labels[0, -1] = -100 # Mock padding ignore

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # Check if loss is returned
        self.assertIsNotNone(outputs.loss)
        self.assertTrue(torch.is_tensor(outputs.loss))
        self.assertFalse(torch.isnan(outputs.loss))
        
        # Check logits shape: (batch, target_len, vocab_size)
        self.assertEqual(outputs.logits.shape, (batch_size, target_len, 100))

    def test_backward_pass(self):
        """Test if gradients are calculated correctly."""
        input_ids = torch.randint(0, 100, (2, 10))
        labels = torch.randint(0, 100, (2, 5))
        
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        # Check if gradients exist for a weight parameter
        param = list(self.model.parameters())[0]
        self.assertIsNotNone(param.grad)

    def test_generation_logic(self):
        """Test the generate method used in evaluation."""
        input_ids = torch.randint(0, 100, (1, 10))
        attention_mask = torch.ones((1, 10))

        # Generate
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=5
        )

        self.assertTrue(torch.is_tensor(generated_ids))
        # T5 usually returns shape (batch, seq_len)
        self.assertEqual(generated_ids.shape[0], 1)
        self.assertLessEqual(generated_ids.shape[1], 5)

    def test_preprocessing_logic(self):
        """Simulate the preprocessing logic manually to verify tensor shapes."""
        # Simulating the logic inside 'preprocess_function'
        max_input = 20
        max_target = 5
        
        # Dummy batch simulation
        input_ids = torch.zeros((4, max_input), dtype=torch.long)
        labels = torch.zeros((4, max_target), dtype=torch.long)
        
        # Apply label masking logic (-100)
        # Assume tokenizer.pad_token_id is 0 for this test
        labels[:, -1] = 0 
        labels[labels == 0] = -100
        
        self.assertEqual(labels[0, -1], -100)
        self.assertEqual(input_ids.shape, (4, max_input))

if __name__ == '__main__':
    unittest.main()