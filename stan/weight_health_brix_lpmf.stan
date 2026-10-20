real weight_health_brix_lpmf(array[] int y, vector mu, vector poor, vector fair,
    vector good, vector serum_brix_mu, real serum_brix_sigma, real serum_brix_nu, 
    vector weightpoor, vector weightfair, vector weightgood, vector weightexcellent,
    real weightsigma, real weightnu, array[] int animal_id, array[] int serum_observed,
    array[] int animal_row, array[] int serum_class, array[] int weight_subset,
    array[] int idx_obs_poor_padded, array[] int idx_obs_fair_padded, 
    array[] int idx_obs_good_padded, array[] int idx_obs_exce_padded, 
    array[] int idx_miss_padded, array[] int idx_obs_weight_poor_padded, 
    array[] int idx_obs_weight_fair_padded, array[] int idx_obs_weight_good_padded,
    array[] int idx_obs_weight_exce_padded, array[] int idx_miss_weight_padded,
    array[] int idx_obs_anim_padded, array[] int idx_miss_anim_padded,
    array[] int counts_padded, array[] real log_exposure, array[] real serum_z, 
    array[] real cut1, array[] real cut2, array[] real cut3, array[] real body_weight) {
  /*
   * For animals with missing serum:
   *
   * joint_delta[a,1] = log likelihood difference:
   *                    Poor - Excellent
   * joint_delta[a,2] = Fair - Excellent
   * joint_delta[a,3] = Good - Excellent
   *
   * The Excellent likelihood itself is accumulated directly
   * into lp, so we only need the three differences here.
   */
  real lp = 0.0;
  
  // recovering indices vectors
  int N = size(y);
  int N_animals = max(animal_id);
  int N_obs_poor = counts_padded[2];
  int N_obs_fair = counts_padded[3];
  int N_obs_good = counts_padded[4];
  int N_obs_exce = counts_padded[5];
  int N_miss = counts_padded[6];
  int N_obs_weight_poor = counts_padded[8];
  int N_obs_weight_fair = counts_padded[9];
  int N_obs_weight_good = counts_padded[10];
  int N_obs_weight_exce = counts_padded[11];
  int N_miss_weight = counts_padded[12];
  int N_obs_anim = counts_padded[13];
  int N_miss_anim = counts_padded[14];
  
  array[N_obs_poor] int idx_obs_poor = idx_obs_poor_padded[1:N_obs_poor];
  array[N_obs_fair] int idx_obs_fair = idx_obs_fair_padded[1:N_obs_fair];
  array[N_obs_good] int idx_obs_good = idx_obs_good_padded[1:N_obs_good];
  array[N_obs_exce] int idx_obs_exce = idx_obs_exce_padded[1:N_obs_exce];
  array[N_miss] int idx_miss = idx_miss_padded[1:N_miss];
  array[N_obs_weight_poor] int idx_obs_weight_poor = idx_obs_weight_poor_padded[1:N_obs_weight_poor];
  array[N_obs_weight_fair] int idx_obs_weight_fair = idx_obs_weight_fair_padded[1:N_obs_weight_fair];
  array[N_obs_weight_good] int idx_obs_weight_good = idx_obs_weight_good_padded[1:N_obs_weight_good];
  array[N_obs_weight_exce] int idx_obs_weight_exce = idx_obs_weight_exce_padded[1:N_obs_weight_exce];
  array[N_miss_weight] int idx_miss_weight = idx_miss_weight_padded[1:N_miss_weight];
  array[N_obs_anim] int idx_obs_anim = idx_obs_anim_padded[1:N_obs_anim];
  array[N_miss_anim] int idx_miss_anim = idx_miss_anim_padded[1:N_miss_anim];
  
  // common/base health contribution
  vector[N] hb = to_vector(log_exposure) + mu;
  /*
   * For every observation the Excellent health likelihood can
   * be used as the common baseline.
   *
   * For observed serum this becomes the actual known-class
   * likelihood below.
   *
   * For missing serum it is always included as the common
   * mixture baseline.
   */
  
  // ----------------------------
  // OBSERVED SERUM BRIX
  // ----------------------------
  
  // observed health likelihood
  lp += poisson_log_lupmf(y[idx_obs_poor] | hb[idx_obs_poor] + poor[idx_obs_poor]);
  lp += poisson_log_lupmf(y[idx_obs_fair] | hb[idx_obs_fair] + fair[idx_obs_fair]);
  lp += poisson_log_lupmf(y[idx_obs_good] | hb[idx_obs_good] + good[idx_obs_good]);
  lp += poisson_log_lupmf(y[idx_obs_exce] | hb[idx_obs_exce]);
  
  // observed weight likelihhod
  lp += student_t_lupdf(body_weight[idx_obs_weight_poor] | weightnu, 
                              weightpoor[idx_obs_weight_poor], weightsigma);
  lp += student_t_lupdf(body_weight[idx_obs_weight_fair] | weightnu, 
                              weightfair[idx_obs_weight_fair], weightsigma);
  lp += student_t_lupdf(body_weight[idx_obs_weight_good] | weightnu, 
                              weightgood[idx_obs_weight_good], weightsigma);
  lp += student_t_lupdf(body_weight[idx_obs_weight_exce] | weightnu, 
                              weightexcellent[idx_obs_weight_exce], weightsigma);

  // observed serum
  lp += student_t_lupdf(serum_z[idx_obs_anim] | serum_brix_nu, 
                              serum_brix_mu[idx_obs_anim], serum_brix_sigma);
  
  // ----------------------------
  // MISSING SERUM BRIX
  // ----------------------------
  array[N_animals] vector[3] joint_delta = rep_array(rep_vector(0.0, 3), N_animals);
  
  // missing health likelihood
  array[N_miss] int y_miss = y[idx_miss];
  vector[N_miss] hb_miss = hb[idx_miss];
    
  lp += poisson_log_lupmf(y_miss | hb_miss);
  
  vector[N_miss] eh_miss = exp(hb_miss);
  vector[N_miss] poor_miss = poor[idx_miss];
  vector[N_miss] fair_miss = fair[idx_miss];
  vector[N_miss] good_miss = good[idx_miss];

  vector[N_miss] expm1_poor = expm1(poor_miss);
  vector[N_miss] expm1_fair = expm1(fair_miss);
  vector[N_miss] expm1_good = expm1(good_miss);
  
  vector[N_miss] y_miss_vec = to_vector(y_miss);

  vector[N_miss] delta_h_poor = y_miss_vec .* poor_miss - eh_miss .* expm1_poor;
  vector[N_miss] delta_h_fair = y_miss_vec .* fair_miss - eh_miss .* expm1_fair;
  vector[N_miss] delta_h_good = y_miss_vec .* good_miss - eh_miss .* expm1_good;


  for (i in 1:N_miss) {
    int n = idx_miss[i];
    int a = animal_id[n];
    joint_delta[a][1] += delta_h_poor[i];
    joint_delta[a][2] += delta_h_fair[i];
    joint_delta[a][3] += delta_h_good[i];
  }
  
  // missing weight likelihhod
  array[N_miss_weight] real w_miss = body_weight[idx_miss_weight];
    vector[N_miss_weight] wexc_miss = weightexcellent[idx_miss_weight];
      
    lp += student_t_lupdf(w_miss | weightnu, wexc_miss, weightsigma);
  
  real inv_weightsigma = 1.0 / weightsigma;
  vector[N_miss_weight] w_vec = to_vector(w_miss);
  vector[N_miss_weight] z_e = (w_vec - wexc_miss) * inv_weightsigma;
  vector[N_miss_weight] z_p = (w_vec - weightpoor[idx_miss_weight]) * inv_weightsigma;
  vector[N_miss_weight] z_f = (w_vec - weightfair[idx_miss_weight]) * inv_weightsigma;
  vector[N_miss_weight] z_g = (w_vec - weightgood[idx_miss_weight]) * inv_weightsigma;

  real inv_weightnu = 1.0 / weightnu;
  vector[N_miss_weight] log1p_z_e = log1p(square(z_e) * inv_weightnu);
  vector[N_miss_weight] log1p_z_p = log1p(square(z_p) * inv_weightnu);
  vector[N_miss_weight] log1p_z_f = log1p(square(z_f) * inv_weightnu);
  vector[N_miss_weight] log1p_z_g = log1p(square(z_g) * inv_weightnu);

  real c = -0.5 * (weightnu + 1.0);
  
  vector[N_miss_weight] delta_w_poor = c * (log1p_z_p - log1p_z_e);
  vector[N_miss_weight] delta_w_fair = c * (log1p_z_f - log1p_z_e);
  vector[N_miss_weight] delta_w_good = c * (log1p_z_g - log1p_z_e);

  for (i in 1:N_miss_weight) {
    int n = idx_miss_weight[i];
    int a = animal_id[n];

    joint_delta[a][1] += delta_w_poor[i];
    joint_delta[a][2] += delta_w_fair[i];
    joint_delta[a][3] += delta_w_good[i];
  }

  // ==============================================================
  // MARGINALIZE MISSING SERUM BRIX
  // ==============================================================

  for (i in 1:N_miss_anim) {
    int n = idx_miss_anim[i];
    int a = animal_id[n];

    // ----------------------------------------------------------
    // Serum class probabilities
    // ----------------------------------------------------------

    real lcdf1 = student_t_lcdf(cut1[n] | serum_brix_nu, serum_brix_mu[n],
                  serum_brix_sigma);
    real lcdf2 = student_t_lcdf(cut2[n] | serum_brix_nu, serum_brix_mu[n],
                  serum_brix_sigma);
    real lcdf3 = student_t_lcdf(cut3[n] | serum_brix_nu, serum_brix_mu[n],
                  serum_brix_sigma);
    vector[4] log_p_class;
    log_p_class[1] = lcdf1 + joint_delta[a][1];
    log_p_class[2] = log_diff_exp(lcdf2, lcdf1) + joint_delta[a][2];
    log_p_class[3] = log_diff_exp(lcdf3, lcdf2) + joint_delta[a][3];
    log_p_class[4] = student_t_lccdf(cut3[n] | serum_brix_nu, serum_brix_mu[n],
                  serum_brix_sigma);

    lp += log_sum_exp(log_p_class);
  }

  return lp;
}