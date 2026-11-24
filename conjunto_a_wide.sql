-- Conjunto A: HSC PDR2 Wide na regiao do campo COSMOS
SELECT
  object_id, ra, dec, isprimary,
  g_cmodel_mag, r_cmodel_mag, i_cmodel_mag, z_cmodel_mag, y_cmodel_mag,
  g_cmodel_magsigma, r_cmodel_magsigma, i_cmodel_magsigma, z_cmodel_magsigma, y_cmodel_magsigma,
  i_extendedness_value,
  a_g, a_r, a_i, a_z, a_y
FROM pdr2_wide.forced
WHERE boxSearch(coord, 148.9, 151.2, 1.0, 3.4)