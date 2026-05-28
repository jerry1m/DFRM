import os
# screw has 5 defect types total; num_anomalies is the total across all 5 classes, equally distributed per class
# os.system("python main_modified.py --flow_arch conditional_flow_model\
#            --gpu 0 --with_fas --data_strategy 0,1 --num_anomalies 50\
#            --not_in_test --exp_name bgad_fas_improved_ano50 --focal_weighting\
#            --pos_beta 0.01 --margin_tau 0.1\
#            --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
#            --meta_epochs 40 --vis --pro --class_name screw --measure_inference ")

# os.system("python main_modified.py --flow_arch conditional_flow_model\
#            --gpu 0 --with_fas --data_strategy 0,1 --num_anomalies 20\
#            --not_in_test --exp_name bgad_fas_improved_ano20 --focal_weighting\
#            --pos_beta 0.01 --margin_tau 0.1\
#            --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
#            --meta_epochs 40 --vis --pro --class_name screw --measure_inference ")

# os.system("python main_modified.py --flow_arch conditional_flow_model\
#            --gpu 0 --with_fas --data_strategy 0,1 --num_anomalies 30\
#            --not_in_test --exp_name bgad_fas_improved_ano30 --focal_weighting\
#            --pos_beta 0.01 --margin_tau 0.1\
#            --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
#            --meta_epochs 40 --vis --pro --class_name screw --measure_inference ")

# os.system("python main_modified.py --flow_arch conditional_flow_model\
#            --gpu 0 --with_fas --data_strategy 0,1 --num_anomalies 40\
#            --not_in_test --exp_name bgad_fas_improved_ano40 --focal_weighting\
#            --pos_beta 0.01 --margin_tau 0.1\
#            --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
#            --meta_epochs 40 --vis --pro --class_name screw --measure_inference ")

# os.system("python main_modified.py --flow_arch conditional_flow_model\
#            --gpu 0 --with_fas --data_strategy 0,1 --num_anomalies 80\
#            --not_in_test --exp_name bgad_fas_improved_ano80 --focal_weighting\
#            --pos_beta 0.01 --margin_tau 0.1\
#            --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
#            --meta_epochs 40 --vis --pro --class_name screw --measure_inference ")

# os.system("python main_modified.py --flow_arch conditional_flow_model\
#            --gpu 0 --with_fas --data_strategy 0,1 --num_anomalies 100\
#            --not_in_test --exp_name bgad_fas_improved_ano100 --focal_weighting\
#            --pos_beta 0.01 --margin_tau 0.1\
#            --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
#            --meta_epochs 40 --vis --pro --class_name screw --measure_inference ")


# os.system("python main_modified.py --flow_arch conditional_flow_model\
#            --gpu 1 --with_fas --data_strategy 0,1,2 --num_anomalies 100\
#            --not_in_test --exp_name bgad_fas_improved_012ano100 --focal_weighting\
#            --pos_beta 0.01 --margin_tau 0.1\
#            --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
#            --meta_epochs 40 --vis --pro --class_name screw --measure_inference ")

# os.system("python main_modified.py --flow_arch conditional_flow_model\
#            --gpu 1 --with_fas --data_strategy 0,1,2 --num_anomalies 20\
#            --not_in_test --exp_name bgad_fas_improved_012ano20 --focal_weighting\
#            --pos_beta 0.01 --margin_tau 0.1\
#            --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
#            --meta_epochs 40 --vis --pro --class_name screw --measure_inference ")

# os.system("python main_modified.py --flow_arch conditional_flow_model\
#            --gpu 1 --with_fas --data_strategy 0,1,2 --num_anomalies 50\
#            --not_in_test --exp_name bgad_fas_improved_012ano50 --focal_weighting\
#            --pos_beta 0.01 --margin_tau 0.1\
#            --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
#            --meta_epochs 40 --vis --pro --class_name screw --measure_inference ")

# os.system("python main_modified.py --flow_arch conditional_flow_model\
#            --gpu 1 --with_fas --data_strategy 0,1,2 --num_anomalies 30\
#            --not_in_test --exp_name bgad_fas_improved_012ano30 --focal_weighting\
#            --pos_beta 0.01 --margin_tau 0.1\
#            --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
#            --meta_epochs 40 --vis --pro --class_name screw --measure_inference ")


# 对于data_strategy仅使用0 即只使用正常样本进行训练,
# 会从测试集的各异常类别直接“每类取 num_anomalies 张”作为训练异常
# os.system("python main_modified.py --flow_arch conditional_flow_model\
#            --gpu 1 --with_fas --data_strategy 0 --num_anomalies 10\
#            --not_in_test --exp_name bgad_fas_improved_0ano10 --focal_weighting\
#            --pos_beta 0.01 --margin_tau 0.1\
#            --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
#            --meta_epochs 40 --vis --pro --class_name screw --measure_inference ")


# 用0,1,2里面的dtd（也就是perlin策略）作为异常源
'''
os.system("python main_modified.py --flow_arch conditional_flow_model\
           --gpu 1 --with_fas --data_strategy 0,1,2 --num_anomalies 10\
           --not_in_test --exp_name bgad_fas_improved_012perlinano10 --focal_weighting\
           --pos_beta 0.01 --margin_tau 0.1 --pseudo_type perlin \
           --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
           --anomaly_source_path /home/luguanghui/PRNet/BGAD-improved/datasets/dtd/images\
           --meta_epochs 40 --vis --pro --class_name screw --measure_inference ")

os.system("python main_modified.py --flow_arch conditional_flow_model\
           --gpu 1 --with_fas --data_strategy 0,1,2 --num_anomalies 40\
           --not_in_test --exp_name bgad_fas_improved_012ano40 --focal_weighting\
           --pos_beta 0.01 --margin_tau 0.1\
           --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
           --meta_epochs 40 --vis --pro --class_name screw --measure_inference ")

os.system("python main_modified.py --flow_arch conditional_flow_model\
           --gpu 1 --with_fas --data_strategy 0,1,2 --num_anomalies 10\
           --not_in_test --exp_name bgad_fas_improved_012ano10 --focal_weighting\
           --pos_beta 0.01 --margin_tau 0.1\
           --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
           --meta_epochs 40 --vis --pro --class_name screw --measure_inference ")

os.system("python main_modified.py --flow_arch conditional_flow_model\
           --gpu 1 --with_fas --data_strategy 0,1,2 --num_anomalies 20\
           --not_in_test --exp_name bgad_fas_improved_012ano20 --focal_weighting\
           --pos_beta 0.01 --margin_tau 0.1\
           --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
           --meta_epochs 40 --vis --pro --class_name screw --measure_inference ")
'''
'''
os.system("python main_modified.py --flow_arch conditional_flow_model --gpu 1 \
  --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection \
  --class_name screw --with_fas --data_strategy 0,1 --num_anomalies 50 \
  --feature_levels 4  --batch_size 32 \
  --exp_name bgad_fl4_01ano50 --focal_weighting --pos_beta 0.01 --margin_tau 0.1\
  --meta_epochs 40 --vis --pro --class_name screw --measure_inference")

os.system("python main_modified.py --flow_arch conditional_flow_model --gpu 1 \
  --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection \
  --class_name screw --with_fas --data_strategy 0,1 --num_anomalies 50 \
  --backbone_arch tf_efficientnet_b7 --feature_levels 3 \
  --batch_size 32 \
  --exp_name bgad_tf_b7_01ano50 --focal_weighting --pos_beta 0.01 --margin_tau 0.1\
  --meta_epochs 40 --vis --pro --class_name screw --measure_inference")

os.system("python main_modified.py --flow_arch conditional_flow_model --gpu 1 \
  --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection \
  --class_name screw --with_fas --data_strategy 0,1 --num_anomalies 50 \
  --backbone_arch tf_efficientnet_b7 --feature_levels 3 \
  --batch_size 32 \
  --exp_name bgad_tf_b7_01ano50_80epo --focal_weighting --pos_beta 0.01 --margin_tau 0.1\
  --meta_epochs 80 --vis --pro --class_name screw --measure_inference")
  '''
'''
# 12.26 先看看b7+fl4的结果 然后还有后面的strong_ops的结果 后续再加上这个增强策略吧

os.system("python main_modified.py --flow_arch conditional_flow_model\
           --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 30\
           --not_in_test --exp_name bgad_tfb7_fl4_01ano30 --focal_weighting\
           --pos_beta 0.01 --margin_tau 0.1\
           --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
           --meta_epochs 40 --vis --pro --class_name screw --measure_inference\
           --backbone_arch tf_efficientnet_b7 --feature_levels 4 \
           --use_seamless_clone \
           --strong_ops 0 \
           --placement_attempts 5 ")

# 增大epochs
os.system("python main_modified.py --flow_arch conditional_flow_model\
           --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 30\
           --not_in_test --exp_name bgad_tfb7_fl4_01ano30_70e --focal_weighting\
           --pos_beta 0.01 --margin_tau 0.1\
           --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
           --meta_epochs 70 --vis --pro --class_name screw --measure_inference\
           --backbone_arch tf_efficientnet_b7 --feature_levels 4 \
           --use_seamless_clone \
           --strong_ops 0 \
           --placement_attempts 5 ")

# 给那两个类多一点训练样本，不额外增强 看看效果  30/5=6 3*6=18
os.system("python main_modified.py --flow_arch conditional_flow_model\
           --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 30\
           --not_in_test --exp_name bgad_tfb7_f14_ano30_0_32 --focal_weighting\
           --pos_beta 0.01 --margin_tau 0.1\
           --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
           --meta_epochs 40 --vis --pro --class_name screw --measure_inference\
           --backbone_arch tf_efficientnet_b7 --feature_levels 4 \
           --prioritized_ano_types thread_side:3,manipulated_front:2\
           --use_seamless_clone \
           --strong_ops 0 \
           --placement_attempts 5 ")
# 增大epochs
os.system("python main_modified.py --flow_arch conditional_flow_model\
           --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 30\
           --not_in_test --exp_name bgad_tfb7_f14_ano30_0_32_70e --focal_weighting\
           --pos_beta 0.01 --margin_tau 0.1\
           --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
           --meta_epochs 70 --vis --pro --class_name screw --measure_inference\
           --backbone_arch tf_efficientnet_b7 --feature_levels 4 \
           --prioritized_ano_types thread_side:3,manipulated_front:2\
           --use_seamless_clone \
           --strong_ops 0 \
           --placement_attempts 5 ")


# 额外增强:tf7b7_fl4_op4 30/5=6 3*6=18
os.system("python main_modified.py --flow_arch conditional_flow_model\
           --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 30\
           --not_in_test --exp_name bgad_b7_fl4_4_33 --focal_weighting\
           --pos_beta 0.01 --margin_tau 0.1\
           --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
           --meta_epochs 40 --vis --pro --class_name screw --measure_inference\
            --prioritized_ano_types thread_side:3,manipulated_front:3\
          --backbone_arch tf_efficientnet_b7 --feature_levels 4 \
          --use_seamless_clone \
          --strong_ops 4 \
          --placement_attempts 5 ")

os.system("python main_modified.py --flow_arch conditional_flow_model\
           --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 30\
           --not_in_test --exp_name bgad_b7_4_33 --focal_weighting\
           --pos_beta 0.01 --margin_tau 0.1\
           --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
           --meta_epochs 40 --vis --pro --class_name screw --measure_inference\
            --prioritized_ano_types thread_side:3,manipulated_front:3\
          --backbone_arch tf_efficientnet_b7 --feature_levels 3 \
          --use_seamless_clone \
          --strong_ops 4 \
          --placement_attempts 5 ")
          
os.system("python main_modified.py --flow_arch conditional_flow_model\
           --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 30\
           --not_in_test --exp_name bgad_b7_fl4_2_33 --focal_weighting\
           --pos_beta 0.01 --margin_tau 0.1\
           --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
           --meta_epochs 40 --vis --pro --class_name screw --measure_inference\
            --prioritized_ano_types thread_side:3,manipulated_front:3\
          --backbone_arch tf_efficientnet_b7 --feature_levels 4 \
          --use_seamless_clone \
          --strong_ops 2 \
          --placement_attempts 5 ")

os.system("python main_modified.py --flow_arch conditional_flow_model\
           --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 30\
           --not_in_test --exp_name bgad_b7_2_33 --focal_weighting\
           --pos_beta 0.01 --margin_tau 0.1\
           --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
           --meta_epochs 40 --vis --pro --class_name screw --measure_inference\
            --prioritized_ano_types thread_side:3,manipulated_front:3\
          --backbone_arch tf_efficientnet_b7 --feature_levels 3 \
          --use_seamless_clone \
          --strong_ops 2 \
          --placement_attempts 5 ")

'''



'''
os.system("python main_modified.py --flow_arch conditional_flow_model\
           --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 30\
           --not_in_test --exp_name bgad_fas_improved_01ano30aug --focal_weighting\
           --pos_beta 0.01 --margin_tau 0.1\
           --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
           --meta_epochs 40 --vis --pro --class_name screw --measure_inference\
            --prioritized_ano_types thread_side:3,manipulated_front:2\
          --use_seamless_clone \
          --strong_ops 6 \
          --placement_attempts 5 ")



os.system("python main_modified.py --flow_arch conditional_flow_model\
           --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 30\
           --not_in_test --exp_name bgad_fas_improved_01ano30aug4 --focal_weighting\
           --pos_beta 0.01 --margin_tau 0.1\
           --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
           --meta_epochs 40 --vis --pro --class_name screw --measure_inference\
            --prioritized_ano_types thread_side:3,manipulated_front:2\
          --use_seamless_clone \
          --strong_ops 4 \
          --placement_attempts 5 ")

os.system("python main_modified.py --flow_arch conditional_flow_model\
           --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 30\
           --not_in_test --exp_name bgad_fas_improved_01ano30aug2 --focal_weighting\
           --pos_beta 0.01 --margin_tau 0.1\
           --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
           --meta_epochs 40 --vis --pro --class_name screw --measure_inference\
            --prioritized_ano_types thread_side:3,manipulated_front:2\
          --use_seamless_clone \
          --strong_ops 2 \
          --placement_attempts 5 ")


# bgad_fas_improved_012ano30aug4_22 后缀代表prioritized_ano_types的倍率
# strong_ops设置为4
os.system("python main_modified.py --flow_arch conditional_flow_model\
           --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 30\
           --not_in_test --exp_name bgad_fas_improved_01ano30aug4_22 --focal_weighting\
           --pos_beta 0.01 --margin_tau 0.1\
           --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
           --meta_epochs 40 --vis --pro --class_name screw --measure_inference\
            --prioritized_ano_types thread_side:2,manipulated_front:2\
          --use_seamless_clone \
          --strong_ops 4 \
          --placement_attempts 5 ")

os.system("python main_modified.py --flow_arch conditional_flow_model\
           --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 30\
           --not_in_test --exp_name bgad_fas_improved_01ano30aug4_1515 --focal_weighting\
           --pos_beta 0.01 --margin_tau 0.1\
           --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
           --meta_epochs 40 --vis --pro --class_name screw --measure_inference\
            --prioritized_ano_types thread_side:1.5,manipulated_front:1.5\
          --use_seamless_clone \
          --strong_ops 4 \
          --placement_attempts 5 ")

os.system("python main_modified.py --flow_arch conditional_flow_model\
           --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 30\
           --not_in_test --exp_name bgad_fas_improved_01ano30aug4_33 --focal_weighting\
           --pos_beta 0.01 --margin_tau 0.1\
           --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
           --meta_epochs 40 --vis --pro --class_name screw --measure_inference\
            --prioritized_ano_types thread_side:3,manipulated_front:3\
          --use_seamless_clone \
          --strong_ops 4 \
          --placement_attempts 5 ")


os.system("python main_modified.py --flow_arch conditional_flow_model\
           --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 20\
           --not_in_test --exp_name bgad_fas_improved_01ano20aug4_44 --focal_weighting\
           --pos_beta 0.01 --margin_tau 0.1\
           --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
           --meta_epochs 40 --vis --pro --class_name screw --measure_inference\
            --prioritized_ano_types thread_side:4,manipulated_front:4\
          --use_seamless_clone \
          --strong_ops 4 \
          --placement_attempts 5 ")

# strong_ops设置为2
os.system("python main_modified.py --flow_arch conditional_flow_model\
           --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 30\
           --not_in_test --exp_name bgad_fas_improved_01ano30au2_22 --focal_weighting\
           --pos_beta 0.01 --margin_tau 0.1\
           --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
           --meta_epochs 40 --vis --pro --class_name screw --measure_inference\
            --prioritized_ano_types thread_side:2,manipulated_front:2\
          --use_seamless_clone \
          --strong_ops 2 \
          --placement_attempts 5 ")

os.system("python main_modified.py --flow_arch conditional_flow_model\
           --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 30\
           --not_in_test --exp_name bgad_fas_improved_01ano30aug2_1515 --focal_weighting\
           --pos_beta 0.01 --margin_tau 0.1\
           --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
           --meta_epochs 40 --vis --pro --class_name screw --measure_inference\
            --prioritized_ano_types thread_side:1.5,manipulated_front:1.5\
          --use_seamless_clone \
          --strong_ops 2 \
          --placement_attempts 5 ")

os.system("python main_modified.py --flow_arch conditional_flow_model\
           --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 30\
           --not_in_test --exp_name bgad_fas_improved_01ano30aug2_33 --focal_weighting\
           --pos_beta 0.01 --margin_tau 0.1\
           --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
           --meta_epochs 40 --vis --pro --class_name screw --measure_inference\
            --prioritized_ano_types thread_side:3,manipulated_front:3\
          --use_seamless_clone \
          --strong_ops 2 \
          --placement_attempts 5 ")


os.system("python main_modified.py --flow_arch conditional_flow_model\
           --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 20\
           --not_in_test --exp_name bgad_fas_improved_01ano20aug2_44 --focal_weighting\
           --pos_beta 0.01 --margin_tau 0.1\
           --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
           --meta_epochs 40 --vis --pro --class_name screw --measure_inference\
            --prioritized_ano_types thread_side:4,manipulated_front:4\
          --use_seamless_clone \
          --strong_ops 2 \
          --placement_attempts 5 ")

'''

# 12.29 经过统计验证，发现b7+fl4+ano50+80epo的结果最好
# os.system("python main_modified.py --flow_arch conditional_flow_model\
#            --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 50\
#            --not_in_test --exp_name bgad_b7_fl4_ano50_80e --focal_weighting\
#            --pos_beta 0.01 --margin_tau 0.1\
#            --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
#            --meta_epochs 80 --vis --pro --class_name screw --measure_inference\
#           --backbone_arch tf_efficientnet_b7 --feature_levels 4 \
#           --use_seamless_clone \
#           --strong_ops 4 \
#           --placement_attempts 5 ")

# os.system("python main_modified.py --flow_arch conditional_flow_model\
#            --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 50\
#            --not_in_test --exp_name bgad_b7_0_1818 --focal_weighting\
#            --pos_beta 0.01 --margin_tau 0.1\
#            --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
#            --meta_epochs 80 --vis --pro --class_name screw --measure_inference\
#             --prioritized_ano_types thread_side:1.8,manipulated_front:1.8\
#           --backbone_arch tf_efficientnet_b7 --feature_levels 4 \
#           --use_seamless_clone \
#           --strong_ops 0 \
#           --placement_attempts 5 ")

# os.system("python main_modified.py --flow_arch conditional_flow_model\
#            --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 50\
#            --not_in_test --exp_name bgad_fl2_ano50 --focal_weighting\
#            --pos_beta 0.01 --margin_tau 0.1\
#            --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
#            --meta_epochs 40 --vis --pro --class_name screw --measure_inference\
#            --feature_levels 2 \
#           --use_seamless_clone \
#           --strong_ops 4 \
#           --placement_attempts 5 ")


# os.system("python main_modified.py \
#   --gpu 1 \
#   --with_fas \
#   --data_strategy 0,1 \
#   --pseudo_type nsa \
#   --anomaly_source_path vis_results/thread_copy_batch/images \
#   --num_anomalies 30 \
#   --not_in_test \
#   --exp_name bgad_finetune_pseudo_screw \
#   --focal_weighting \
#   --checkpoint output/bgad_fas_improved_01ano30aug4_1515/mvtec_tf_efficientnet_b6_conditional_flow_model_screw.pt \
#   --lr 5e-05 \
#   --meta_epochs 40 \
#   --sub_epochs 4 \
#   --pos_beta 0.01 \
#   --margin_tau 0.1 \
#   --class_name screw \
#   --measure_inference \
#   --vis --pro \
#   --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection")

# os.system("python main_modified.py \
#   --gpu 1 \
#   --with_fas \
#   --data_strategy 0,1 \
#   --pseudo_type nsa \
#   --anomaly_source_path vis_results/thread_copy_batch_1/images \
#   --num_anomalies 50 \
#   --not_in_test \
#   --exp_name bgad_finetune_pseudo_screw_0123 \
#   --focal_weighting \
#   --checkpoint output/bgad_fas_improved_01ano50_0123/mvtec_tf_efficientnet_b6_conditional_flow_model_screw.pt \
#   --lr 5e-05 \
#   --meta_epochs 40 \
#   --sub_epochs 4 \
#   --pos_beta 0.01 \
#   --margin_tau 0.1 \
#   --class_name screw \
#   --measure_inference \
#   --vis --pro \
#   --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection")

# 生成缺陷
# os.system("python generate_thread.py   --anomaly_dir datasets/mvtec_anomaly_detection/screw/test\
#      --anomaly_mask_dir datasets/mvtec_anomaly_detection/screw/ground_truth   \
#      --normal_dir datasets/mvtec_anomaly_detection/screw/train/good   --fg_root fg_mask/screw \
#      --out_dir_cp vis_results/thread_copy_batch_1   --scale_range 0.7 1.1   --rot_deg -10 10   --max_attempts 40   --min_pixels 200 ")

# os.system("python main_modified.py --flow_arch conditional_flow_model\
#            --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 5\
#            --not_in_test --exp_name bgad_fas_improved_01ano5_01_0316 --focal_weighting\
#            --pos_beta 0.01 --margin_tau 0.1\
#            --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
#            --meta_epochs 25 --vis --pro --class_name none --measure_inference ")

# os.system("python main_modified.py --flow_arch conditional_flow_model\
#            --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 5\
#            --not_in_test --exp_name bgad_fas_improved_01ano5_01_quaternion --focal_weighting\
#            --pos_beta 0.01 --margin_tau 0.1\
#            --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
#            --meta_epochs 25 --vis --pro --class_name none --measure_inference  --backbone_arch quaternion_cnn")

# os.system("python main_modified.py --flow_arch conditional_flow_model\
#            --gpu 0 --with_fas --data_strategy 0,1 --num_anomalies 10\
#            --not_in_test --exp_name bgad_fas_improved_ano10_screw_ori --focal_weighting\
#            --pos_beta 0.01 --margin_tau 0.1\
#            --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
#            --meta_epochs 25 --vis --pro --class_name screw --measure_inference ")

# os.system("python main_modified.py --flow_arch conditional_flow_model\
#            --gpu 0 --with_fas --data_strategy 0,1 --num_anomalies 5\
#            --not_in_test --exp_name bgad_fas_improved_bottle --focal_weighting\
#            --pos_beta 0.01 --margin_tau 0.1\
#            --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
#            --meta_epochs 25 --vis --pro --class_name bottle --measure_inference ")

# os.system("python main_modified.py --flow_arch conditional_flow_model\
#            --gpu 1 --with_fas --data_strategy 0,1 --num_anomalies 10\
#            --not_in_test --exp_name bgad_fas_improved_all_0321 --focal_weighting\
#            --pos_beta 0.01 --margin_tau 0.1 --backbone_arch resnet_se  \
#            --data_path /home/luguanghui/PRNet/BGAD-improved/datasets/mvtec_anomaly_detection\
#            --meta_epochs 25 --vis --pro --class_name none --measure_inference ")


# os.system(
#     "python visualize_se_channel_heatmaps.py "
#     "--image-dir /home/luguanghui/PRNet/BGAD-improved-2-2-2/datasets/mvtec_anomaly_detection/screw/test/thread_top "
#     "--checkpoint output/bgad_fas_improved_screw/weights/mvtec_tf_efficientnet_b6_conditional_flow_model_screw.pt "
#     "--outdir se_heatmaps234_screw_ts "
#     "--topk 3 "
    
#     "--layers 4 "
#     "--backbone_arch resnet_se "
#     "--backbone_base resnet50"
# )

os.system("python main_modified.py \
  --dataset visa \
  --data_path /home/luguanghui/PRNet/VisA \
  --with_fas \
  --flow_arch conditional_flow_model \
  --backbone_arch tf_efficientnet_b6 \
  --feature_levels 3 \
  --batch_size 32 \
  --meta_epochs 25 \
  --gpu 0 \
  --class_name none\
  --num_anomalies 10 \
  --vis --pro \
  --not_in_test \
  --exp_name visa_none_10ano ")


# os.system("python main_modified.py \
#   --dataset btad \
#   --data_path /home/luguanghui/PRNet/BTech_Dataset_transformed \
#   --with_fas \
#   --flow_arch conditional_flow_model \
#   --backbone_arch tf_efficientnet_b6 \
#   --feature_levels 3 \
#   --batch_size 32 \
#   --meta_epochs 25 \
#   --gpu 1 \
#   --class_name none\
#   --num_anomalies 10 \
#   --vis --pro \
#   --not_in_test \
#   --exp_name btad_none_10ano ")
