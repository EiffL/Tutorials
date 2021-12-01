"""hsc_photoz dataset."""
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# TODO(hsc_photoz): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
"""

# TODO(hsc_photoz): BibTeX citation
_CITATION = """
"""

# Attributes for each galaxy
_attrs = ['a_g',
 'a_r',
 'a_i',
 'a_z',
 'a_y',
 'g_extendedness_value',
 'r_extendedness_value',
 'i_extendedness_value',
 'z_extendedness_value',
 'y_extendedness_value',
 'g_localbackground_flux',
 'r_localbackground_flux',
 'i_localbackground_flux',
 'z_localbackground_flux',
 'y_localbackground_flux',
 'g_cmodel_flux',
 'g_cmodel_fluxsigma',
 'g_cmodel_exp_flux',
 'g_cmodel_exp_fluxsigma',
 'g_cmodel_dev_flux',
 'g_cmodel_dev_fluxsigma',
 'r_cmodel_flux',
 'r_cmodel_fluxsigma',
 'r_cmodel_exp_flux',
 'r_cmodel_exp_fluxsigma',
 'r_cmodel_dev_flux',
 'r_cmodel_dev_fluxsigma',
 'i_cmodel_flux',
 'i_cmodel_fluxsigma',
 'i_cmodel_exp_flux',
 'i_cmodel_exp_fluxsigma',
 'i_cmodel_dev_flux',
 'i_cmodel_dev_fluxsigma',
 'z_cmodel_flux',
 'z_cmodel_fluxsigma',
 'z_cmodel_exp_flux',
 'z_cmodel_exp_fluxsigma',
 'z_cmodel_dev_flux',
 'z_cmodel_dev_fluxsigma',
 'y_cmodel_flux',
 'y_cmodel_fluxsigma',
 'y_cmodel_exp_flux',
 'y_cmodel_exp_fluxsigma',
 'y_cmodel_dev_flux',
 'y_cmodel_dev_fluxsigma',
 'g_cmodel_mag',
 'g_cmodel_magsigma',
 'g_cmodel_exp_mag',
 'g_cmodel_exp_magsigma',
 'g_cmodel_dev_mag',
 'g_cmodel_dev_magsigma',
 'r_cmodel_mag',
 'r_cmodel_magsigma',
 'r_cmodel_exp_mag',
 'r_cmodel_exp_magsigma',
 'r_cmodel_dev_mag',
 'r_cmodel_dev_magsigma',
 'i_cmodel_mag',
 'i_cmodel_magsigma',
 'i_cmodel_exp_mag',
 'i_cmodel_exp_magsigma',
 'i_cmodel_dev_mag',
 'i_cmodel_dev_magsigma',
 'z_cmodel_mag',
 'z_cmodel_magsigma',
 'z_cmodel_exp_mag',
 'z_cmodel_exp_magsigma',
 'z_cmodel_dev_mag',
 'z_cmodel_dev_magsigma',
 'y_cmodel_mag',
 'y_cmodel_magsigma',
 'y_cmodel_exp_mag',
 'y_cmodel_exp_magsigma',
 'y_cmodel_dev_mag',
 'y_cmodel_dev_magsigma',
 'g_sdssshape_shape11',
 'g_sdssshape_shape12',
 'g_sdssshape_shape22',
 'g_sdssshape_psf_shape11',
 'g_sdssshape_psf_shape12',
 'g_sdssshape_psf_shape22',
 'r_sdssshape_shape11',
 'r_sdssshape_shape12',
 'r_sdssshape_shape22',
 'r_sdssshape_psf_shape11',
 'r_sdssshape_psf_shape12',
 'r_sdssshape_psf_shape22',
 'i_sdssshape_shape11',
 'i_sdssshape_shape12',
 'i_sdssshape_shape22',
 'i_sdssshape_psf_shape11',
 'i_sdssshape_psf_shape12',
 'i_sdssshape_psf_shape22',
 'z_sdssshape_shape11',
 'z_sdssshape_shape12',
 'z_sdssshape_shape22',
 'z_sdssshape_psf_shape11',
 'z_sdssshape_psf_shape12',
 'z_sdssshape_psf_shape22',
 'y_sdssshape_shape11',
 'y_sdssshape_shape12',
 'y_sdssshape_shape22',
 'y_sdssshape_psf_shape11',
 'y_sdssshape_psf_shape12',
 'y_sdssshape_psf_shape22',
 'd_pos',
 'd_mag',
 'specz_ra',
 'specz_dec',
 'specz_redshift',
 'specz_redshift_err',
 'specz_mag_i']

def stack_bands(cutout, target_size=64):
  import numpy as np

  filters = ['HSC-G', 'HSC-R', 'HSC-I', 'HSC-Z', 'HSC-Y']

  # Retrieve the cutouts in all the bands
  im = [cutout[f]['HDU0']['DATA'][:] for f in filters]

  # Stack all bands in one array
  im_size = min([min(i.shape) for i in im])
  im = np.stack([i[:im_size, :im_size] for i in im], axis=-1).astype('float32')

  # Resize image to target size
  centh = im.shape[0]/2
  centw = im.shape[1]/2
  lh, rh = int(centh-target_size/2), int(centh+target_size/2)
  lw, rw = int(centw-target_size/2), int(centw+target_size/2)
  cropped = im[lh:rh, lw:rw, :]
  assert cropped.shape[0]==target_size and cropped.shape[1]==target_size, f"Wrong size! Still {cropped.shape}"
  return cropped

class HscPhotoz(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for hsc_photoz dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(hsc_photoz): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
    'image': tfds.features.Tensor(shape=(64, 64, 5), dtype=tf.float32),
    'attrs': {k: tf.float32 for k in _attrs}
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'attrs/specz_redshift'),
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    catalog_path = dl_manager.download('https://storage.googleapis.com/ahw2019/hsc_photoz/data/catalog.fits')
    cutouts_path = dl_manager.download('https://storage.googleapis.com/ahw2019/hsc_photoz/data/cutouts.hdf')
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={'catalog_path': catalog_path,
                        'cutouts_path': cutouts_path},
        ),
    ]

  def _generate_examples(self,catalog_path, cutouts_path):
    """Yields examples."""
    from astropy.table import Table
    import h5py

    # Loading the data that was downloaded at the previous step
    catalog = Table.read(catalog_path)
    cutouts = h5py.File(cutouts_path, 'r')

    # Go through the examples
    for object_id in catalog['object_id']:
      row = catalog[catalog['object_id'] == object_id]
      cutout = cutouts[str(object_id)]

      # extract image from cutouts
      im = stack_bands(cutout)

      yield object_id, {'image': im, 
                        'attrs':{k: np.asscalar(row[k]) for k in _attrs}}
