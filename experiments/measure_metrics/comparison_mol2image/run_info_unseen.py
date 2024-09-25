RUN_INFO = {
            'IMPA': ['/home/icb/alessandro.palma/environment/IMPA/IMPA/config_hydra/config/REBUTTAL_bbbc021_large_all.yaml', 
                            '/home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/experiments/20240902_fc588378-f1f0-4cfb-84f5-92ed48f31a44_bbbc021_unannotated_large', 
                            '000200'],
            
            'mol2image': ['/home/icb/alessandro.palma/environment/IMPA/mol2image/mol2img_scripts/normal/mol2image_96.yaml',
                       [{"type": "cond_proglowmpn", "n_flow": 32, "n_block": 1, "in_channel": 3, "img_size": 128, "n_cond": 150, 
                           'pretrained': '/home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/mol2image/20240902_128/checkpoint/063500_nets.ckpt'},
                       {"type": "cond_proglowmpn", "n_flow": 32, "n_block": 1, "in_channel": 3, "img_size": 64, "n_cond": 150, 
                        'pretrained': '/home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/mol2image/20240901_64/checkpoint/049500_nets.ckpt'},
                       {"type": "cond_glowmpn", "n_flow": 32, "n_block": 1, "in_channel": 3, "img_size": 32,  "n_cond": 150, 
                        'pretrained': '/home/icb/alessandro.palma/environment/IMPA/IMPA/project_folder/mol2image/20240825_32/checkpoint/093000_nets.ckpt'}]]
            }
