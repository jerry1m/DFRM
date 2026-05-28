import os
# os.system("python test.py   --dataset mvtec   --class_name screw   \
#           --backbone_arch tf_efficientnet_b6   --flow_arch conditional_flow_model   \
#           --feature_levels 3   --inp_size 256   --checkpoint output/bgad_fas_improved_01ano50   \
#           --output_dir vis_results0260114    --exp_name bgad_fas_improved_01ano50   --vis   --gpu 0 --pro")  

# os.system("python test.py   --dataset mvtec   --class_name screw   \
#           --backbone_arch tf_efficientnet_b7   --flow_arch conditional_flow_model   \
#           --feature_levels 4   --inp_size 256   --checkpoint output/bgad_tfb7_fl4_01ano30_70e   \
#           --output_dir vis_results20260114    --exp_name bgad_tfb7_fl4_01ano30_70e   --vis   --gpu 0 --pro")  

# os.system("python test_pseudo_anomaly.py \
#          --images_dir datasets/mvtec_anomaly_detection/screw/test/thread_side \
#          --mask_dir datasets/mvtec_anomaly_detection/screw/ground_truth/thread_side \
#          --checkpoint_dir output/bgad_fas_improved_01ano30aug4_1515  \
#          --class_name screw   --out_dir vis_results/pseudo_thread_copy_run_TS  \
#          --device cuda   --threshold_mode gt   --threshold_candidates 100")

# os.system("python test_pseudo_anomaly.py \
#          --images_dir datasets/mvtec_anomaly_detection/screw/test/scratch_head \
#          --mask_dir datasets/mvtec_anomaly_detection/screw/ground_truth/scratch_head \
#          --checkpoint_dir output/bgad_fas_improved_01ano30aug4_1515  \
#          --class_name screw   --out_dir vis_results/pseudo_thread_copy_run_SH  \
#          --device cuda   --threshold_mode gt   --threshold_candidates 100")

# os.system("python test_pseudo_anomaly.py \
#          --images_dir datasets/mvtec_anomaly_detection/screw/test/scratch_neck \
#          --mask_dir datasets/mvtec_anomaly_detection/screw/ground_truth/scratch_neck \
#          --checkpoint_dir output/bgad_fas_improved_01ano30aug4_1515  \
#          --class_name screw   --out_dir vis_results/pseudo_thread_copy_run_SN  \
#          --device cuda   --threshold_mode gt   --threshold_candidates 100")

# os.system("python test_pseudo_anomaly.py \
#          --images_dir datasets/mvtec_anomaly_detection/screw/test/thread_top \
#          --mask_dir datasets/mvtec_anomaly_detection/screw/ground_truth/thread_top \
#          --checkpoint_dir output/bgad_fas_improved_01ano30aug4_1515  \
#          --class_name screw   --out_dir vis_results/pseudo_thread_copy_run_TT  \
#          --device cuda   --threshold_mode gt   --threshold_candidates 100")

# os.system("python test_pseudo_anomaly.py \
#          --images_dir vis_results/thread_copy_batch/images \
#          --mask_dir vis_results/thread_copy_batch/masks \
#          --checkpoint_dir output/bgad_fas_improved_01ano30aug4_1515  \
#          --class_name screw   --out_dir vis_results/pseudo_thread_copy_run_116  \
#          --device cuda   --threshold_mode gt   --threshold_candidates 100")

# os.system("python test_pseudo_anomaly.py \
#          --images_dir datasets/mvtec_anomaly_detection/screw/test/good \
#          --checkpoint_dir output/bgad_fas_improved_01ano30aug4_1515  \
#          --class_name screw   --out_dir vis_results/pseudo_thread_copy_run_GO  \
#          --device cuda   --threshold_mode gt   --threshold_candidates 100")


# os.system("python test_pseudo_anomaly.py \
#          --images_dir datasets/mvtec_anomaly_detection/screw/test/manipulated_front \
#          --mask_dir vis_results/thread_copy_batch/masks \
#          --checkpoint_dir output/bgad_fas_improved_01ano30aug4_1515  \
#          --class_name screw   --out_dir vis_results/pseudo_thread_copy_run_MF_2  \
#          --device cuda   --threshold_mode quantile   --threshold_candidates 100")


# os.system("python test.py   --dataset mvtec   --class_name screw   \
#           --backbone_arch tf_efficientnet_b6   --flow_arch conditional_flow_model   \
#              --inp_size 256   --checkpoint output/bgad_finetune_pseudo_screw   \
#           --output_dir vis_results20260119    --exp_name bgad_finetune_pseudo_screw   --vis   --gpu 0 --pro")  

# os.system("python test.py   --dataset mvtec   --class_name screw   \
#           --backbone_arch tf_efficientnet_b6   --flow_arch conditional_flow_model   \
#              --inp_size 256   --checkpoint output/bgad_fas_improved_ano10_screw_ori/weights   \
#           --output_dir vis_results20260122_3    --exp_name bgad_screw_test   --vis   --gpu 1 --pro")  
 

# os.system("python test.py   --dataset mvtec   --class_name none   \
#           --backbone_arch resnet_se   --flow_arch conditional_flow_model   \
#              --inp_size 256   --checkpoint output/bgad_fas_improved_all_0321/weights   \
#           --output_dir vis_results0321    --exp_name bgad_all_test   --vis   --gpu 1 --pro")  

os.system("python test.py   --dataset btad   --class_name 03   \
          --backbone_arch resnet_se   --flow_arch conditional_flow_model   \
          --inp_size 256   --checkpoint /home/luguanghui/PRNet/BGAD-improved-2-2-2/output/cross_mvtec2btad_weights   \  
          --data_path /home/luguanghui/PRNet/BTech_Dataset_transformed \
          --output_dir cross_mvtecbottle2btad03    --exp_name cross_mvtecbottle2btad03   --vis --gpu 1 --pro")