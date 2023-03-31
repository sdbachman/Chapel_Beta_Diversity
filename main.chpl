//use png;
use distance_mask;
use dissimilarity;
use beta_diversity;
use IO_Module;
use IO;
use BlockDist;
use Time;
use AutoMath;
use LinearAlgebra;

/* Command line arguments. */
config const in_array : string;                /* name of binary file to read */
config const dissimilarity_file : string;    /* name of the file with dissimilarity coefficients */
config const window_size : real(32);                  /* the desired area of the neighborhood (in meters^2) */
config const dx : real(32);                      /* the resolution of the raster image (in meters) */
config const out_file : string;
config const cols : int = 1;
config const rows : int = 1;

proc convolve_and_calculate(Image: [] int(8), centerPoints : ?, LeftMaskDomain : ?, CenterMaskDomain : ?, RightMaskDomain : ?, dissimilarity : [] real(32), Output: [] real(32), d_size : int, Mask_Size : int,  t: stopwatch) : [] {

  // This 'eps' makes sure that we differentiate between land points (zero) and ocean points (nonzero), even
  // if the beta diversity at the ocean point is zero.
  param eps = 0.00001;

  //writeln("On ", here.id, " centerPoints is: ", centerPoints);

  var first_point = centerPoints.first[1];
  var last_point = centerPoints.last[1];

  forall center in centerPoints[..,first_point] {

      // Calculate masks and beta diversity for leftmost point in subdomain
      var B_left: [0..(d_size-1)] real(32) = 0;
      var B_center: [0..(d_size-1)] real(32) = 0;
      var B_right: [0..(d_size-1)] real(32) = 0;

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
      var B_old = B;

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
        var num_habitat_pixels = (+ reduce B[1..(d_size-2)]) : real(32);
        var habitat_frac = num_habitat_pixels / Mask_Size;

        var P = B / num_habitat_pixels;

        var beta = + reduce (dissimilarity * outer(P,P));
        Output[center,first_point] = (habitat_frac * beta) : real(32);
      }

      for point in (first_point+1)..last_point do {

        B_right = 0;
        for m in RightMaskDomain do {
          var tmp = Image[(center,point) + m];
          B_right[tmp] = B_right[tmp] + 1;
        }
        B = B_old + B_right - B_left;
        B_old = B;
        B_left = 0;

        // Update B_left
        for m in LeftMaskDomain do {
          var tmp = Image[(center,point) + m];
          B_left[tmp] = B_left[tmp] + 1;
        }

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
          var num_habitat_pixels = (+ reduce B[1..(d_size-2)]) : real(32);
          var habitat_frac = num_habitat_pixels / Mask_Size;

          var P = B / num_habitat_pixels;

          var beta = + reduce (dissimilarity * outer(P,P));
          Output[center,point] = (habitat_frac * beta + eps) : real(32);
        }
      }
  }

  writeln("Elapsed time on ", here.name, ": ", t.elapsed(), " seconds for domain ", centerPoints);

}


proc main(args: [] string) {

  var t : stopwatch;
  t.start();

  // Gather input variables from command line
  const radius = (sqrt(window_size) / 2) : int;
  const nx = (radius / dx) : int;
  writeln("Distance circle has a radius of ", nx, " points.");

  const ImageSpace = {0..<rows, 0..<cols};
  var Image : [ImageSpace] int(8);

  // Read in array
  var f = open(in_array, iomode.r);
  var r = f.reader(kind=ionative);

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
  var OutputArray : [D] real(32);

  // Create NetCDF
  var varid : int;
  CreateNetCDF(out_file, ImageSpace, varid);

  writeln("Elapsed time at start of coforall loop: ", t.elapsed(), " seconds.");

  writeln("Starting coforall loop.");

//////////////////////////////////////////////////////////////////////////

  coforall loc in Locales do on loc {

    const loc_d_size = d_size;
    const loc_Mask_Size = Mask_Size;

    var locD = D.localSubdomain();
    var locD_plus = locD.expand(offset);
    var locImage : [locD_plus] int(8);

    // Read in array
    var f = open(in_array, iomode.r);
    var first_point = locD_plus.first[0]*locD_plus.shape[1] + locD_plus.first[1];
    var r = f.reader(kind=ionative, region=first_point..);

    for i in locD_plus.first[0]..locD_plus.last[0] {
      for j in locD_plus.first[1]..locD_plus.last[1] {
        var tmp : int(8);
        r.readBinary(tmp);
        locImage[i,j] = tmp;
      }
    }
    r.close();

    const locLeftMaskDomain = LeftMask.domain;
    const locCenterMaskDomain = CenterMask.domain;
    const locRightMaskDomain = RightMask.domain;

    const locDissDomain = d_domain;
    const locDiss : [locDissDomain] dissimilarity.eltType = dissimilarity;

    convolve_and_calculate(locImage, D.localSubdomain(), locLeftMaskDomain, locCenterMaskDomain, locRightMaskDomain, locDiss, OutputArray, loc_d_size, loc_Mask_Size, t);
  }

  writeln("Elapsed time to finish coforall loop: ", t.elapsed(), " seconds.");

  WriteOutput(out_file, OutputArray, ImageSpace, varid, offset);

  writeln("Elapsed time to write NetCDF: ", t.elapsed(), " seconds.");

}

