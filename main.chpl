use png;
use distance_mask;
use dissimilarity;
use beta_diversity;
use BlockDist;
use Time;
use AutoMath;
use LinearAlgebra;

/* Command line arguments. */
config const inname : string;                /* name of PNG file to read */
config const outname : string;               /* name of PNG file to write at the end */
config const dissimilarity_file : string;    /* name of the file with dissimilarity coefficients */
config const window_size : real;                  /* the desired area of the neighborhood (in meters^2) */
config const dx : real;                      /* the resolution of the raster image (in meters) */

proc convolve_and_calculate(Image: [] int(64), centerPoints : ?, LeftMaskDomain : ?, CenterMaskDomain : ?, RightMaskDomain : ?, dissimilarity : [] real, Output: [] real, d_size : int, Mask_Size : int,  t: stopwatch) : [] {

  // This 'eps' makes sure that we differentiate between land points (zero) and ocean points (nonzero), even
  // if the beta diversity at the ocean point is zero.
  param eps = 0.00001;

  var first_point = centerPoints.first[1];
  var last_point = centerPoints.last[1];

  forall center in centerPoints[..,first_point] {

      // Calculate masks and beta diversity for leftmost point in subdomain
      var B_left: [0..(d_size-1)] real = 0;
      var B_center: [0..(d_size-1)] real = 0;
      var B_right: [0..(d_size-1)] real = 0;

      for m in LeftMaskDomain do {
        var tmp = Image[(center,first_point) + m];
        B_left[tmp] = B_left[tmp] + 1;
      }
      for m in CenterMaskDomain do {
        var tmp = Image[(center,first_point) + m];
        B_center[tmp] = B_center[tmp] + 1;
      }
      for m in RightMaskDomain do {
        var tmp = Image[(center,first_point) + m];
        B_right[tmp] = B_right[tmp] + 1;
      }

      var B = B_left + B_center + B_right;
      var B_center_prev = B_center;

      // If we are over land, return zero
      if (Image[center,first_point] == 0) {
        Output[center,first_point] = 0.0;
      }
      // If we are over deep water, return a different number so we can color it differently
      else if (Image[center, first_point] == (d_size-1)) {
        Output[center,first_point] = -999.0;
      }
      // If we are on a reef point, calculate beta diversity
      else {
        var num_habitat_pixels = (+ reduce B[1..(d_size-2)]) : real;
        var habitat_frac = num_habitat_pixels / Mask_Size;

        var P = B / num_habitat_pixels;

        var beta = + reduce (dissimilarity * outer(P,P));
        Output[center,first_point] = habitat_frac * beta;
      }

      for point in (first_point+1)..last_point do {
        B_center = B_center_prev + B_right - B_left;
        B_center_prev = B_center;
        B_left = 0;
        B_right = 0;

        for m in LeftMaskDomain do {
          var tmp = Image[(center,point) + m];
          B_left[tmp] = B_left[tmp] + 1;
        }
        for m in RightMaskDomain do {
          var tmp = Image[(center,point) + m];
          B_right[tmp] = B_right[tmp] + 1;
        }
        B = B_left + B_center + B_right;

        // If we are over land, return zero
        if (Image[center,point] == 0) {
          Output[center,point] = 0.0;
        }
        // If we are over deep water, return a different number so we can color it differently
        else if (Image[center,point] == (d_size-1)) {
          Output[center,point] = -999.0;
        }
        // If we are on a reef point, calculate beta diversity
        else {
          var num_habitat_pixels = (+ reduce B[1..(d_size-2)]) : real;
          var habitat_frac = num_habitat_pixels / Mask_Size;

          var P = B / num_habitat_pixels;

          var beta = + reduce (dissimilarity * outer(P,P));
          Output[center,point] = habitat_frac * beta + eps;
        }
      }
  }

  writeln("Elapsed time on ", here.name, ": ", t.elapsed(), " seconds for domain ", centerPoints);

}


proc main(args: [] string) {

  var t : stopwatch;
  t.start();

  // Gather input variables from command line
  /*
  const input_file = args[1];
  const dissimilarity_file = args[2];
  const window_size : real = (args[3] : real);
  const dx : real = (args[4] : real);
  */
  const radius = (sqrt(window_size) / 2) : int;
  const nx = (radius / dx) : int;
  writeln("Distance circle has a radius of ", nx, " points.");

  // Read in PNG
  var (rgb_ptr, Image) = load_PNG_into_array(inname, t);
  const ImageSpace = Image.domain;
  writeln("ImageSpace is ", ImageSpace);
  writeln("Elapsed time to read into array: ", t.elapsed(), " seconds.");

  // Read in dissimilarity coefficients
  var (dissimilarity, d_size) = ReadArray(dissimilarity_file);

  // Shift the domain so that it starts at 0
  var d_domain = dissimilarity.domain.translate(-1);

  // Create distance mask
  var (LeftMask, CenterMask, RightMask, Mask_Size) = create_distance_mask(radius, dx, nx);

  // Create Block distribution of interior of PNG
  const offset = nx; // maybe needs to be +1 to account for truncation?
  const Inner = ImageSpace.expand(-offset);
  const myTargetLocales = reshape(Locales, {1..Locales.size, 1..1});
  const D = Inner dmapped Block(Inner, targetLocales=myTargetLocales);
  var OutputArray : [D] real;

  writeln("Elapsed time at start of coforall loop: ", t.elapsed(), " seconds.");

  writeln("Starting coforall loop.");

  coforall loc in Locales do on loc {

    // If I put "create_distance_mask" inside this loop I need to declare local copies of these variables,
    // otherwise it seems like Chapel will have to do a ton
    // of cross-locale calls to access these variables. This seems to double the amount of time it
    // takes to run through the coforall loop for all non-head locales!

    const loc_d_size = d_size;
    const loc_Mask_Size = Mask_Size;

    const locImageDomain = Image.domain;
    const locImage : [locImageDomain] Image.eltType = Image;

    const locLeftMaskDomain = LeftMask.domain;
    const locCenterMaskDomain = CenterMask.domain;
    const locRightMaskDomain = RightMask.domain;

    const locDissDomain = d_domain;
    const locDiss : [locDissDomain] dissimilarity.eltType = dissimilarity;

    convolve_and_calculate(locImage, D.localSubdomain(), locLeftMaskDomain, locCenterMaskDomain, locRightMaskDomain, locDiss, OutputArray, loc_d_size, loc_Mask_Size, t);
  }


  writeln("Elapsed time to finish coforall loop: ", t.elapsed(), " seconds.");

  // Gather back to the head node
  var GatheredArray : [Inner] real;
  GatheredArray = OutputArray;

  write_array_to_PNG(outname, GatheredArray, rgb_ptr, t);

  writeln("Elapsed time to write PNG: ", t.elapsed(), " seconds.");
}

