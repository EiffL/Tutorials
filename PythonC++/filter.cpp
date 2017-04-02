#include "filter.hpp"


np::ndarray filter(np::ndarray &im){

    int n_x = im.shape(0);
    int n_y = im.shape(1);

    // Build output array with same dimension and same type as input
    p::tuple shape = p::make_tuple(n_x, n_y);
    np::ndarray im_out = np::zeros(shape, im.get_dtype());

    for(int x=1; x < n_x-1; x++){
      for(int y=1; y < n_y-1; y++){
          im_out[y, x] = im[y, x+1] - im[y, x-1];
      }
    }

    return im_out;
}
