-- Conjunto B (espectroscopico), adaptado do apendice do GalaxiesML (Do et al. 2024)
-- Mudanca em relacao ao artigo: incluimos isprimary, absorcao (a_*) e d_pos como colunas
SELECT
  object_id,
  ra,
  dec,
  isprimary,
  g_cmodel_mag,
  r_cmodel_mag,
  i_cmodel_mag,
  z_cmodel_mag,
  y_cmodel_mag,
  g_cmodel_magsigma,
  r_cmodel_magsigma,
  i_cmodel_magsigma,
  z_cmodel_magsigma,
  y_cmodel_magsigma,
  a_g,
  a_r,
  a_i,
  a_z,
  a_y,
  specz_name,
  specz_ra,
  specz_dec,
  specz_redshift,
  specz_redshift_err,
  specz_flag_homogeneous,
  specz_mag_i,
  d_pos
FROM pdr2_wide.forced
LEFT JOIN pdr2_wide.forced2 USING (object_id)
LEFT JOIN pdr2_wide.specz USING (object_id)
WHERE
  NOT g_sdsscentroid_flag
  AND NOT r_sdsscentroid_flag
  AND NOT i_sdsscentroid_flag
  AND NOT z_sdsscentroid_flag
  AND NOT y_sdsscentroid_flag
  AND NOT g_pixelflags_interpolatedcenter
  AND NOT g_pixelflags_saturatedcenter
  AND NOT g_pixelflags_edge
  AND NOT g_pixelflags_crcenter
  AND NOT g_pixelflags_bad
  AND NOT r_pixelflags_interpolatedcenter
  AND NOT r_pixelflags_saturatedcenter
  AND NOT r_pixelflags_edge
  AND NOT r_pixelflags_crcenter
  AND NOT r_pixelflags_bad
  AND NOT i_pixelflags_interpolatedcenter
  AND NOT i_pixelflags_saturatedcenter
  AND NOT i_pixelflags_edge
  AND NOT i_pixelflags_crcenter
  AND NOT i_pixelflags_bad
  AND NOT z_pixelflags_interpolatedcenter
  AND NOT z_pixelflags_saturatedcenter
  AND NOT z_pixelflags_edge
  AND NOT z_pixelflags_crcenter
  AND NOT z_pixelflags_bad
  AND NOT y_pixelflags_interpolatedcenter
  AND NOT y_pixelflags_saturatedcenter
  AND NOT y_pixelflags_edge
  AND NOT y_pixelflags_crcenter
  AND NOT y_pixelflags_bad
  AND NOT g_cmodel_flag
  AND NOT r_cmodel_flag
  AND NOT i_cmodel_flag
  AND NOT z_cmodel_flag
  AND NOT y_cmodel_flag
  AND specz_redshift > 0
  AND specz_redshift_err > 0
  AND specz_redshift_err < 1
  AND specz_redshift < 9.999