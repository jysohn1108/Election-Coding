# Here we have command lines for each figure (averaged out over 3 or 4 trials)


# Fig.4(a),(d): n=5, b=1 
python main.py --gpu_num 7 --trial_idx 1 --deterministic True --redundancy 3.8 --num_nodes 5 --num_Byz_nodes 1 --batch_size 120 #> noiseless_trial_2_n15_b3_r3_batch_120_log 2>&1 # deterministic(r=3.8)
#python main.py --gpu_num 6 --trial_idx 1 --redundancy 2.5 --num_nodes 5 --num_Byz_nodes 1 --batch_size 120 #> noiseless_trial_2_n15_b3_r3_batch_120_log 2>&1 # Bern(r=2.5)
#python main.py --gpu_num 7 --trial_idx 1 --deterministic True --redundancy 1.0 --num_nodes 5 --num_Byz_nodes 1 --batch_size 120 #> noiseless_trial_2_n15_b3_r3_batch_120_log 2>&1 # SignSGD-MV (r=1.0)
#python main.py --gpu_num 4 --trial_idx 1 --deterministic True --redundancy 1.0 --num_nodes 5 --num_Byz_nodes 0 --batch_size 120 #> noiseless_trial_2_n15_b3_r3_batch_120_log 2>&1 # No-Attack

# Fig.4(b),(e): n=9, b=2
#python main.py --gpu_num 7 --trial_idx 1 --redundancy 3.0 --num_nodes 9 --num_Byz_nodes 2 --batch_size 576 # Bern(r=3)
#python main.py --gpu_num 4 --trial_idx 1 --redundancy 2.0 --num_nodes 9 --num_Byz_nodes 2 --batch_size 576 # Bern(r=2)
#python main.py --gpu_num 3 --trial_idx 2 --deterministic True --num_nodes 9 --num_Byz_nodes 2 --batch_size 576 # SignSGD-MV(r=1)
#python main.py --gpu_num 3 --trial_idx 1 --deterministic True --num_nodes 9 --num_Byz_nodes 0 --batch_size 576 # No-Attack

# Fig.4(c),(f): n=15, b=3
#python main.py --gpu_num 4 --trial_idx 1 --redundancy 3.0 --num_nodes 15 --num_Byz_nodes 3 --batch_size 240 > n15_b3_r3_batch_240_log 2>&1 # Bern(r=3)
#python main.py --gpu_num 5 --trial_idx 1 --redundancy 2.0 --num_nodes 15 --num_Byz_nodes 3 --batch_size 240 > n15_b3_r2_batch_240_log 2>&1 # Bern(r=2)
#python main.py --gpu_num 6 --trial_idx 1 --deterministic True --redundancy 1.0 --num_nodes 15 --num_Byz_nodes 3 --batch_size 240 > n15_b3_r1_batch_240_log 2>&1 # SignSGD-MV (r=1.0)
#python main.py --gpu_num 7 --trial_idx 1 --deterministic True --redundancy 1.0 --num_nodes 15 --num_Byz_nodes 0 --batch_size 240 > n15_b0_r1_batch_240_log 2>&1 # No-Attack



# Fig.5(a): n=5, b=2
#python main.py --gpu_num 1 --trial_idx 5 --deterministic True --redundancy 3.8 --num_nodes 5 --num_Byz_nodes 2 --batch_size 240 > n5_b2_r3_8_t5_batch_240_log 2>&1 # Deterministic(r=3.8)
#python main.py --gpu_num 2 --trial_idx 5 --redundancy 2.5 --num_nodes 5 --num_Byz_nodes 2 --batch_size 240 > n5_b2_r2_5_t5_batch_240_log 2>&1 # Bern(r=2.5)
#python main.py --gpu_num 3 --trial_idx 5 --deterministic True --redundancy 1.0 --num_nodes 5 --num_Byz_nodes 2 --batch_size 240 > n5_b2_r1_t5_batch_240_log 2>&1 # SignSGD-MV (r=1.0)
#python main.py --gpu_num 7 --trial_idx 1 --deterministic True --redundancy 1.0 --num_nodes 5 --num_Byz_nodes 0 --batch_size 240 > n5_b0_r1_batch_240_log 2>&1 # No-Attack

# Fig.5(b): n=9, b=3
#python main.py --gpu_num 4 --trial_idx 1 --redundancy 3.0 --num_nodes 9 --num_Byz_nodes 3 --batch_size 126 > n9_b3_r3_batch_126_log 2>&1 # Bern(r=3)
#python main.py --gpu_num 3 --trial_idx 2 --redundancy 2.0 --num_nodes 9 --num_Byz_nodes 3 --batch_size 126 > n9_b3_r2_t2_batch_126_log 2>&1  # Bern(r=2)
#python main.py --gpu_num 4 --trial_idx 2 --deterministic True --num_nodes 9 --num_Byz_nodes 3 --batch_size 126 > n9_b3_r1_t2_batch_126_log 2>&1 # SignSGD-MV(r=1)
#python main.py --gpu_num 7 --trial_idx 1 --deterministic True --num_nodes 9 --num_Byz_nodes 0 --batch_size 126 > n9_b0_r1_batch_126_log 2>&1 # No-Attack

# Fig.5(c): n=15, b=6
#python main.py --gpu_num 5 --trial_idx 2 --redundancy 3.0 --num_nodes 15 --num_Byz_nodes 6 --batch_size 240 > n15_b6_r3_t2_batch_240_log 2>&1 # Bern(r=3)
#python main.py --gpu_num 5 --trial_idx 2 --redundancy 2.0 --num_nodes 15 --num_Byz_nodes 6 --batch_size 240 > n15_b6_r2_t2_batch_240_log 2>&1 # Bern(r=2)
#python main.py --gpu_num 7 --trial_idx 2 --deterministic True --redundancy 1.0 --num_nodes 15 --num_Byz_nodes 6 --batch_size 240 > n15_b6_r1_t2_batch_240_log 2>&1 # SignSGD-MV (r=1.0)
#python main.py --gpu_num 7 --trial_idx 1 --deterministic True --redundancy 1.0 --num_nodes 15 --num_Byz_nodes 0 --batch_size 240 > n15_b0_r1_batch_240_log 2>&1 # No-Attack



## Fig.A.1: n=5, b=2
#python main.py --gpu_num 4 --trial_idx 1 --deterministic True --redundancy 3.8 --num_nodes 5 --num_Byz_nodes 2 --batch_size 240 > n5_b2_r3_8_batch_240_log 2>&1 # Deterministic(r=3.8)
#python main.py --gpu_num 7 --trial_idx 1 --deterministic True --redundancy 1.0 --num_nodes 5 --num_Byz_nodes 0 --batch_size 240 > n5_b0_r1_batch_240_log 2>&1 # No-Attack


## Fig.A.2: n=9, b=4
#python main.py --gpu_num 4 --trial_idx 1 --redundancy 3.0 --num_nodes 9 --num_Byz_nodes 4 --batch_size 126 > n9_b4_r3_t1_batch_126_log 2>&1 # Bern(r=3)
#python main.py --gpu_num 5 --trial_idx 1 --deterministic True --num_nodes 9 --num_Byz_nodes 4 --batch_size 126 > n9_b4_r1_t1_batch_126_log 2>&1 # SignSGD-MV(r=1)


## Fig.A.3(a): n=5, b=2
#python main.py --gpu_num 7 --trial_idx 4 --redundancy 2.5 --num_nodes 5 --num_Byz_nodes 2 --batch_size 120 > n5_b2_r2_5_t4_batch_120_log 2>&1 # Bern(r=2.5, \rho=0.5, r_{eff} = 1.25)
#python main.py --gpu_num 5 --trial_idx 1 --deterministic True --redundancy 1.0 --num_nodes 5 --num_Byz_nodes 2 --batch_size 240 > n5_b2_r1_batch_240_log 2>&1 # SignSGD-MV (r=1.0)


## Fig.A.3(b) n=15, b=6
#python main.py --gpu_num 2 --trial_idx 1 --redundancy 3.0 --num_nodes 15 --num_Byz_nodes 6 --batch_size 120 > n15_b6_r3_t1_batch_120_log 2>&1 # Bern(r=3)
#python main.py --gpu_num 3 --trial_idx 1 --deterministic True --redundancy 1.0 --num_nodes 15 --num_Byz_nodes 6 --batch_size 240 > n15_b6_r1_t1_batch_240_log 2>&1 # SignSGD-MV (r=1.0)


## Fig.A.4(a): ResNet-50 (n=15, b=6)
#python main.py --network ResNet50 --gpu_num 2 --trial_idx 1 --deterministic True --redundancy 1.0 --num_nodes 15 --num_Byz_nodes 6 --batch_size 960 > ResNet50_n15_b6_t1_r1_batch_960_log 2>&1 # SignSGD-MV (r=1.0)
#python main.py --network ResNet50 --gpu_num 7 --trial_idx 3 --redundancy 3.0 --num_nodes 15 --num_Byz_nodes 6 --batch_size 480 > ResNet50_n15_b6_t3_r3_batch_480_log 2>&1 # Bern(r=3, \rho=0.5, r_{eff}=1.5)
#python main.py --network ResNet50 --gpu_num 5 --trial_idx 2 --redundancy 2.0 --num_nodes 15 --num_Byz_nodes 6 --batch_size 960 > ResNet50_n15_b6_t2_r2_batch_960_log 2>&1 # Bern(r=2)
#python main.py --network ResNet50 --gpu_num 4 --num_epochs 100 --trial_idx 1 --redundancy 3.0 --num_nodes 15 --num_Byz_nodes 6 --batch_size 960 > ResNet50_n15_b6_t1_r3_batch_960_log 2>&1 # Bern(r=3)
#python main.py --network ResNet50 --gpu_num 7 --trial_idx 1 --deterministic True --redundancy 1.0 --num_nodes 15 --num_Byz_nodes 0 --batch_size 960 > ResNet50_n15_b0_t1_r1_batch_960_log 2>&1 # No-Attack


## Fig.A.4(b): Noisy Computation (n=15, b=6)
#python main.py --noisy_gradient True --variance 1e-4 --gpu_num 4 --trial_idx 1 --deterministic True --redundancy 1.0 --num_nodes 15 --num_Byz_nodes 6 --batch_size 240 > noisy_gradient_variance_1e-4_n15_b6_r1_t1_batch_240_log 2>&1
#python main.py --noisy_gradient True --variance 1e-4 --gpu_num 2 --trial_idx 4 --redundancy 3.0 --num_nodes 15 --num_Byz_nodes 6 --batch_size 240 > noisy_gradient_variance_1e-4_n15_b6_r3_t4_batch_240_log 2>&1


## Fig.A.4(c): Compare with median (n=15, b=3 or b=6)
#python main.py --gpu_num 5 --trial_idx 1 --redundancy 2.0 --num_nodes 15 --num_Byz_nodes 3 --batch_size 240 > n15_b3_r2_t1_batch_240_log 2>&1 # Bern(r=2), b=3
#python main.py --gpu_num 5 --trial_idx 1 --redundancy 2.0 --num_nodes 15 --num_Byz_nodes 6 --batch_size 240 > n15_b6_r2_t1_batch_240_log 2>&1 # Bern(r=2), b=6
#python main.py --gpu_num 5 --trial_idx 1 --redundancy 3.0 --num_nodes 15 --num_Byz_nodes 6 --batch_size 120 > n15_b6_r3_t1_batch_120_log 2>&1 # Bern(r=3, \rho=0.5, r_{eff}=1.5), b=6
#python main.py --gpu_num 5 --coord_median True --trial_idx 1 --num_nodes 15 --num_Byz_nodes 3 --batch_size 240 #> coordinatewise_median_n15_b3_t1_batch_240_log 2>&1 # b=3, coord-wise med.
#python main.py --gpu_num 5 --coord_median True --trial_idx 1 --num_nodes 15 --num_Byz_nodes 6 --batch_size 240 #> coordinatewise_median_n15_b6_t1_batch_240_log 2>&1 # b=6, coord-wise med.


