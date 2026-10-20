real health_brix_lpmf(array[] int y, vector mu, vector poor, vector fair,
vector good, vector serum_brix_mu, real serum_brix_sigma, real serum_brix_nu,
array[] int animal_id, array[] int serum_observed,
array[] int animal_row, array[] int serum_class,
array[] int idx_obs_poor_padded, array[] int idx_obs_fair_padded, 
array[] int idx_obs_good_padded, array[] int idx_obs_exce_padded, 
array[] int idx_miss_padded, array[] int idx_obs_anim_padded, 
array[] int idx_miss_anim_padded, array[] int counts_padded,
array[] real log_exposure, array[] real serum_z,
array[] real cut1, array[] real cut2, array[] real cut3) {
  
  int N = size(y);
  int N_animals = max(animal_id);
  real lp = 0; // total likelihood

  // mu contains predictors shared across all serum brix classes
  vector[N] base = to_vector(log_exposure) + mu;
  
  // creating arrays for observed and missing values
  int N_obs_poor = counts_padded[2];
  int N_obs_fair = counts_padded[3];
  int N_obs_good = counts_padded[4];
  int N_obs_exce = counts_padded[5];
  int N_miss = counts_padded[6];
  int N_obs_anim = counts_padded[13];
  int N_miss_animals = counts_padded[14];
  int N_serum_obs_animals = counts_padded[15];

  array[N_obs_poor] int idx_obs_poor = idx_obs_poor_padded[1:N_obs_poor];
  array[N_obs_fair] int idx_obs_fair = idx_obs_fair_padded[1:N_obs_fair];
  array[N_obs_good] int idx_obs_good = idx_obs_good_padded[1:N_obs_good];
  array[N_obs_exce] int idx_obs_exce = idx_obs_exce_padded[1:N_obs_exce];
  array[N_miss] int idx_miss = idx_miss_padded[1:N_miss];
  array[N_obs_anim] int idx_obs_anim = idx_obs_anim_padded[1:N_obs_anim];
  array[N_miss_animals] int idx_miss_anim = idx_miss_anim_padded[1:N_miss_animals];
  
  // -------------------------------------
  // observed serum brix
  // -------------------------------------
  
  // health likelihood for observed brix
  lp += poisson_log_lupmf(y[idx_obs_poor] | base[idx_obs_poor] + poor[idx_obs_poor]);
  lp += poisson_log_lupmf(y[idx_obs_fair] | base[idx_obs_fair] + fair[idx_obs_fair]);
  lp += poisson_log_lupmf(y[idx_obs_good] | base[idx_obs_good] + good[idx_obs_good]);
  lp += poisson_log_lupmf(y[idx_obs_exce] | base[idx_obs_exce]);
  
  // serum likelihood for observed brix
  lp += student_t_lupdf(serum_z[idx_obs_anim] | serum_brix_nu, serum_brix_mu[idx_obs_anim], serum_brix_sigma);
  
  
  // -------------------------------------
  // missing serum brix
  // -------------------------------------
  
  // For animals with missing Serum Brix we need all four possible class likelihoods.
  // But, we are considering the excellent class as the baseline. So, all other classes
  // are on top of that (we only need differences)
  array[N_animals] vector[3] joint_delta = rep_array(rep_vector(0.0, 3), N_animals);
  
  
  array[N_miss] int y_miss = y[idx_miss];
  vector[N_miss] base_miss = base[idx_miss];
  // baseline (excellent)
  lp += poisson_log_lupmf(y_miss | base_miss);
  
  vector[N_miss] poor_miss = poor[idx_miss];
  vector[N_miss] fair_miss = fair[idx_miss];
  vector[N_miss] good_miss = good[idx_miss];

  vector[N_miss] exp_base_miss = exp(base_miss);
  vector[N_miss] expm1_poor_miss = expm1(poor_miss);
  vector[N_miss] expm1_fair_miss = expm1(fair_miss);
  vector[N_miss] expm1_good_miss = expm1(good_miss);
  
  vector[N_miss] y_miss_vec = to_vector(y_miss);
  
  vector[N_miss] delta_h_poor = y_miss_vec .* poor_miss - exp_base_miss .* expm1_poor_miss;
  vector[N_miss] delta_h_fair = y_miss_vec .* fair_miss - exp_base_miss .* expm1_fair_miss;
  vector[N_miss] delta_h_good = y_miss_vec .* good_miss - exp_base_miss .* expm1_good_miss;

  for (i in 1:N_miss) {
    int a = animal_id[idx_miss[i]];
    joint_delta[a][1] += delta_h_poor[i];
    joint_delta[a][2] += delta_h_fair[i];
    joint_delta[a][3] += delta_h_good[i];
  }
  
  // marginalizing serum brix
  for (i in 1:N_miss_animals) {
    int n = idx_miss_anim[i];
    int a = animal_id[n];
      
    real lcdf_cut1 = student_t_lcdf(cut1[n] | serum_brix_nu, serum_brix_mu[n], serum_brix_sigma);
    real lcdf_cut2 = student_t_lcdf(cut2[n] | serum_brix_nu, serum_brix_mu[n], serum_brix_sigma);
    real lcdf_cut3 = student_t_lcdf(cut3[n] | serum_brix_nu, serum_brix_mu[n], serum_brix_sigma);
    
    
    vector[4] class_lp;
    class_lp[1] = lcdf_cut1 + joint_delta[a][1];
    class_lp[2] = log_diff_exp(lcdf_cut2, lcdf_cut1) + joint_delta[a][2];
    class_lp[3] = log_diff_exp(lcdf_cut3, lcdf_cut2) + joint_delta[a][3];
    class_lp[4] = student_t_lccdf(cut3[n] | serum_brix_nu, serum_brix_mu[n], serum_brix_sigma);
      
    lp += log_sum_exp(class_lp);
  }
  
  return lp;
}