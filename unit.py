import unittest

class TestFormatReward(unittest.TestCase):
    def setUp(self):
        self.obj = ChessGRPOTrainer(model, reference, optimizer)  # Replace with the actual class containing compute_format_reward

    def test(self):
        occurences = {start_think_index : 1 , 1 : 1, end_index :1}
        last_occurence = {start_think_index : 0,  1 : 3, end_index :4}
        print(self.obj.compute_format_reward(occurences, last_occurence))
        self.assertEqual(self.obj.compute_format_reward(occurences, last_occurence), -5)

if __name__ == "__main__":
    from grpo import ChessGRPOTrainer
    import torch
    from model import GPT, GPTConfig
    config = GPTConfig()
    config.vocab_size = 1929
    config.block_size = 256


    model = GPT(config).to("cuda")
    reference = GPT(config).to("cuda")
    from data.vocab import policy_index
    from data.parse import dir_iterator
    dir_path = "/media/maxime/Crucial X8/GitRefactored/ParrotChess/pros_pgn"
    gen = dir_iterator(dir_path,triple = True)
    #load weights 
    #model.load_state_dict(torch.load("fine_tune/checkpoint_step_fine_tune_10000.pt"))
    #reference.load_state_dict(torch.load("fine_tune/checkpoint_step_fine_tune_10000.pt"))

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-7)
    # Initialize trainer
    start_think_index = policy_index.index("<thinking>")
    end_think_index = policy_index.index("</thinking>")
    end_variation_index = policy_index.index("end_variation")
    end_index = policy_index.index("end")

    unittest.main()
