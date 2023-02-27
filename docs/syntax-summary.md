# Syntax summary

* access objects: `<(object|database)>.<object>...`: `data.l1single_spectra.ob`
* access attributes: `<object>.<attribute>` or `<object>['attribute']`: `spectrum.snr` or `spectrum['snr']`
* identifying by id: `<object[plural]>[<id[str|int|float]>]`: `data.obs[3443]`
* identifying by ids: `<object[plural]>[<id[str|int|float]>, ...]`: `data.obs[3443, 4421]`
* filtering by mask: `<object|attribute>[<attribute[bool]>]`: `spectra[spectra.snr > 10]`