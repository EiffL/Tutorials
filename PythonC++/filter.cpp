#include "filter.hpp"


np::ndarray filter(np::ndarray &im, filter_type f_type){

    int n_x = im.shape(0);
    int n_y = im.shape(1);

    int stride_x = im.strides(0);
    int stride_y = im.strides(1);

    // Build output array with same dimension and same type as input
    p::tuple shape = p::make_tuple(n_x, n_y);
    np::ndarray im_out = np::zeros(shape, im.get_dtype());

    // Objects to easily access the data in the arrays
    NumPyArrayData<double> im_data(s);
    NumPyArrayData<double> im_out_data(phi);


    // Applying filter to image
    for(int x=0; x < n_x; x++){
      for(int y=0; y < n_y; y++){
        im_out[y, x] =  im[y-1, x-1] + im[y,   x] + im[y-1, x+1]
                    + im[y,   x-1] - 8.*im[y,x] + im[y,   x+1]
                    + im[y+1, x-1] + im[y+1, x] + im[y+1, x+1];
      }
    }

    return im_out;
}
