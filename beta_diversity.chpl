module beta_diversity {

proc calc_beta_diversity_old(Image: [] int(64), center : ?, Mask : [] bool, MaskDomain : ?, dissimilarity : [] real, d_size : int) {

      var B: [1..(d_size-2)] real = 0;
      for m in MaskDomain do {
        var tmp = Image[center + m] * Mask[m];
        B[tmp] = B[tmp] + 1;
      }

      var num_habitat_pixels = + reduce B;
      var num_habitats : real = 0;
      for i in B.domain do {
        if B[i] > 0 {
           num_habitats = num_habitats+1.0;
        }
      }

      // If there is only one habitat, just return zero
      if (num_habitats == 1.0) {
        return 0.0;
      }
      else {

        var habitat_types = [i in B.domain] if B[i] > 0 then i;
        var D : real = 0;
        for i in habitat_types {
          for j in habitat_types {
             D = D + 0.5*dissimilarity[i,j];
          }
        }
        var first_term = log10(D);

        var P = B / num_habitat_pixels;
        var logP = log(P);
        for i in P.domain do if P[i] == 0.0 then logP[i] = 0.0;
        var second_term = + reduce ((P * logP) / (-log(num_habitats)));

        return (first_term * second_term);
      } // else
} // proc

proc calc_beta_diversity_new(Image: [] int(64), center : ?, Mask : [] bool, MaskDomain : ?, dissimilarity : [] real, d_size : int, Mask_size : int) {

      var B: [0..(d_size-1)] real = 0;
      for m in MaskDomain do {
        var tmp = Image[center + m] * Mask[m];
        B[tmp] = B[tmp] + 1;
      }

      var num_habitat_pixels = (+ reduce B[1..(d_size-2)]) : real;
      var habitat_frac = num_habitat_pixels / Mask_size;

      var P = B / num_habitat_pixels;

      var beta = + reduce (dissimilarity * outer(P,P));

      return (habitat_frac * beta);
} // proc

}
