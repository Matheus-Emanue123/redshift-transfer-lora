SELECT
  count(*)              AS total,
  count(g_cmodel_mag)   AS com_g,
  count(r_cmodel_mag)   AS com_r,
  count(i_cmodel_mag)   AS com_i,
  count(z_cmodel_mag)   AS com_z,
  count(y_cmodel_mag)   AS com_y
FROM pdr2_wide.forced
WHERE coneSearch(coord, 150.12, 2.21, 300)