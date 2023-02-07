use NetCDF.C_NetCDF;

proc WriteOutput(ref arr_in: [?D] real, ImageSpace : ?, offset : int) {

  /* IDs for the netCDF file, dimensions, and variables. */
    var ncid, x_dimid, y_dimid : c_int;
    var x_varid, y_varid : c_int;
    var varid : c_int;

    var ndims : int = 2;
    var dimids: [0..#ndims] c_int;

    var shape = ImageSpace.shape;

    /* Need to do this, otherwise the NetCDF image will come out upside down */

    var arr_tmp : [D] real;
    for (i,j) in D {
      arr_tmp[i,j] = arr_in[D.last[0]+offset-i,j];
    }

    var arr_out : [ImageSpace] real = -999;
    arr_out[D] = arr_tmp;

    var yo : [1..arr_out.shape[0]] real;
    var xo : [1..arr_out.shape[1]] real;
    for i in 1..arr_out.shape[0] {
      yo[i] = i;
    }
    for i in 1..arr_out.shape[1] {
      xo[i] = i;
    }

    var yName = "y";
    var xName = "x";

  /* Create the file. */
    extern proc nc_create(path : c_string, cmode : c_int, ncidp : c_ptr(c_int)) : c_int;
    nc_create( ("out.nc").c_str(), NC_CLOBBER, c_ptrTo(ncid));

  /* Define the dimensions. The record dimension is defined to have
     unlimited length - it can grow as needed. In this example it is
     the time dimension.*/
    extern proc nc_def_dim(ncid : c_int, name : c_string, len : c_size_t, idp : c_ptr(c_int)) : c_int;
    nc_def_dim(ncid, yName.c_str(), shape[0] : c_size_t, y_dimid);
    nc_def_dim(ncid, xName.c_str(), shape[1] : c_size_t, x_dimid);

  /* Define the coordinate variables. */
    extern proc nc_def_var(ncid : c_int, name : c_string, xtype : nc_type, ndims : c_int, dimidsp : c_ptr(c_int), varidp : c_ptr(c_int)) : c_int;
      nc_def_var(ncid, yName.c_str(), NC_DOUBLE, 1 : c_int, c_ptrTo(y_dimid), c_ptrTo(y_varid));
      nc_def_var(ncid, xName.c_str(), NC_DOUBLE, 1 : c_int, c_ptrTo(x_dimid), c_ptrTo(x_varid));

  /* Assign units attributes to coordinate variables. */
  //  att_text = "meters";
  //  nc_put_att_text(ncid, y_varid, "units".c_str(), att_text.numBytes : c_size_t, att_text.c_str());
  //  att_text = "meters";
  //  nc_put_att_text(ncid, x_varid, "units".c_str(), att_text.numBytes : c_size_t, att_text.c_str());

  /* The dimids array is used to pass the dimids of the dimensions of
     the netCDF variables. In C, the unlimited dimension must come first on the list of dimids. */
    dimids[0] = y_dimid;
    dimids[1] = x_dimid;

  /* Define the netCDF variable. */
    nc_def_var(ncid, "beta_diversity".c_str(), NC_DOUBLE, ndims : c_int, c_ptrTo(dimids[0]), c_ptrTo(varid));

  /* Assign units attributes to the netCDF variables. */

    var maxval = max reduce(arr_out);
    var valid_range = "(0, " + (maxval : string) + ")";
    //nc_put_att_text(ncid, varid, "units".c_str(), units.numBytes : c_size_t, units.c_str());
    nc_put_att_text(ncid, varid, "Range".c_str(), valid_range.numBytes : c_size_t, valid_range.c_str());

    //int nc_def_var_fill(int ncid, int varid, int no_fill, const void* fill_value)
    extern proc nc_def_var_fill(ncid : c_int, varid : c_int, no_fill : c_int, fill_value : c_ptr(c_double));
    var fv = -999 : real;
    nc_def_var_fill(ncid, varid, 0, c_ptrTo(fv));

  /* End define mode. */
    nc_enddef(ncid);

  /* Write the coordinate variable data. */
    extern proc nc_put_var_double(ncid : c_int, varid : c_int, op : c_ptr(c_double)) : c_int;
    nc_put_var_double(ncid, y_varid, c_ptrTo(yo[0]));
    nc_put_var_double(ncid, x_varid, c_ptrTo(xo[0]));

  /* Determine where to start reading file, and how many elements to read */
  // Start specifies a hyperslab.  It expects an array of dimension sizes
   // var start = tuplify(ImageSpace.localSubdomain().first);
  // Count specifies a hyperslab.  It expects an array of dimension sizes
   // var count = tuplify(ImageSPace.localSubdomain().shape);

    var start = (0,0);
    var count = ImageSpace.shape;

    var start_c : [0..#start.size] c_size_t;
    var count_c : [0..#count.size] c_size_t;
    for i in 0..count.size {
      count_c[i] = count[i] : c_size_t;
    }

    extern proc nc_put_vara_double(ncid : c_int, varid : c_int, startp : c_ptr(c_size_t), countp : c_ptr(c_size_t), op : c_ptr(c_double)) : c_int;
    nc_put_vara_double(ncid, varid, c_ptrTo(start_c), c_ptrTo(count_c), c_ptrTo(arr_out[start]));

    nc_close(ncid);

}

inline proc tuplify(x) {
  if isTuple(x) then return x; else return (x,);
}

