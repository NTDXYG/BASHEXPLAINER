from second_stage_train.model import BASHEXPLAINER

# 加载微调过的模型
model = BASHEXPLAINER(codebert_path = 'D:\\new_idea\\Final\\model\\codebert', decoder_layers = 6, fix_encoder = True, beam_size = 10,
                         max_source_length = 64, max_target_length = 64, load_model_path = '../pretrained_model/first_stage/pytorch_model.bin',
                         l2_norm=True, fusion=True)

# 模型训练
model.train(train_filename ='../data/second_stage_data/train.csv', train_batch_size = 64, num_train_epochs = 30, learning_rate = 5e-5,
            do_eval = True, dev_filename ='../data/second_stage_data/valid.csv', eval_batch_size = 64, output_dir ='../pretrained_model/first_stage')

# 加载微调过的模型
model = BASHEXPLAINER(codebert_path = '/data/home/yangguang/models/codebert-base', decoder_layers = 6, fix_encoder = True, beam_size = 10,
                         max_source_length = 64, max_target_length = 64, load_model_path = '../pretrained_model/second_stage/pytorch_model.bin')

# 模型测试
model.test(test_filename ='../data/second_stage_data/test.csv', test_batch_size = 16, output_dir ='result')