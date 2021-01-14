from pathlib import Path

import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy import units as u, constants
import matplotlib.pyplot as plt


class BasicSpectra:
    ulam = u.AA
    uadu = u.adu

    def __init__(self, wvls, adu, aduivar, calibration):
        self.ucal = calibration.unit
        self.uivaradu = 1 / self.uadu / self.uadu
        self.uflam = self.ucal * u.adu
        self.uivarflam = 1 / (self.uflam ** 2)
        self._wvls = wvls.to(self.ulam)
        self._adu = np.atleast_2d(adu).to(self.uadu)
        self._aduivar = np.atleast_2d(aduivar).to(self.uivaradu)
        self._calibration = np.atleast_2d(calibration).to(self.ucal)
        if self._calibration.shape != self._aduivar.shape != self._adu.shape:
            raise TypeError("Shapes do not match")
        self._flux = (self._adu * self._calibration).to(self.uflam)
        self._fluxivar = (self._aduivar / (self._calibration ** 2)).to(self.uivarflam)
        self._fluxerr = 1 / np.sqrt(self._fluxivar)
        self._aduerr = 1 / np.sqrt(self._aduivar)
        self._freqs = (constants.c / wvls).to(u.Hz)

    @property
    def wvls(self):
        return self._wvls

    @property
    def adu(self):
        return self._adu

    @property
    def aduivar(self):
        return self._aduivar

    @property
    def calibration(self):
        return self._calibration

    @property
    def flux(self):
        return self._flux

    @property
    def fluxivar(self):
        return self._fluxivar

    @property
    def fluxerr(self):
        return self._fluxerr

    @property
    def aduerr(self):
        return self._aduerr

    def __len__(self):
        return self.flux.shape[0]

    def __repr__(self):
        return f"<{self.__class__.__name__}({len(self)} in {self.uflam})"

    @property
    def snr(self):
        return (self.adu / self.aduerr).value

    def plot(self, ax=None, fluxtype='flux', xunit='lam', legend=False, **kwargs):
        return self[:].plot(ax, fluxtype, xunit, legend, **kwargs)

    @classmethod
    def from_weave_obfits(cls, hdus, flux_hdu: int, ivar_hdu: int, calib_hdu: int):
        wvl_start = hdus[flux_hdu].header['CRVAL1']
        wvl_inc = hdus[flux_hdu].header['CD1_1']
        flux = hdus[flux_hdu].data
        invflux = hdus[ivar_hdu].data
        calibration = hdus[calib_hdu].data
        wvl_end = (flux.shape[1] * wvl_inc) + wvl_start
        wvls = np.arange(wvl_start, wvl_end, wvl_inc) * u.AA
        cal = calibration * u.erg / u.s / u.cm / u.cm / u.adu / u.AA
        return cls(wvls, flux * u.adu, invflux / u.adu / u.adu, cal)



class BasicIndexedSpectra(BasicSpectra):
    def __init__(self, host_spectra, index):
        self.host_spectra = host_spectra
        self.index = index

    @property
    def uadu(self):
        return self.host_spectra.uadu
    @property
    def uflam(self):
        return self.host_spectra.uflam
    @property
    def ucal(self):
        return self.host_spectra.ucal
    @property
    def uwvl(self):
        return self.host_spectra.uwvl
    @property
    def uivaradu(self):
        return self.host_spectra.uivaradu
    @property
    def uivarflam(self):
        return self.host_spectra.uivarflam

    @property
    def wvls(self):
        return self.host_spectra.wvls

    @property
    def freqs(self):
        return self.host_spectra.freqs

    @property
    def names(self):
        return self.host_spectra.names[self.index]

    @property
    def names_array(self):
        return self.host_spectra.names_array[self.index]

    @property
    def adu(self):
        return self.host_spectra.adu[self.index]

    @property
    def aduivar(self):
        return self.host_spectra.aduivar[self.index]

    @property
    def calibration(self):
        return self.host_spectra.calibration[self.index]

    @property
    def flux(self):
        return self.host_spectra.flux[self.index]

    @property
    def fluxivar(self):
        return self.host_spectra.fluxivar[self.index]

    @property
    def fluxerr(self):
        return self.host_spectra.fluxerr[self.index]

    @property
    def aduerr(self):
        return self.host_spectra.aduerr[self.index]

    @property
    def arms(self):
        return list(map(str, np.asarray(self.host_spectra.arms)[self.index]))

    @property
    def resolutions(self):
        return list(map(str, np.asarray(self.host_spectra.resolutions)[self.index]))

    def squeeze(self):
        if len(self) == 1:
            return BasicSpectrum(self)
        else:
            return self

    def plot(self, ax=None, fluxtype='flux', xunit='lam', legend=False, **kwargs):
        if fluxtype == 'flux':
            flux = self.flux.to(self.uflam)
        elif fluxtype == 'counts' or fluxtype == 'adu':
            flux = self.adu
        else:
            raise ValueError(f"{fluxtype} is not a valid fluxtype")
        if xunit == 'lam':
            x = self.wvls
            xx = fr"$\lambda / {self.wvls.unit.to_string('latex').strip('$')}$"
        elif xunit == 'freq':
            x = self.freqs
            xx = fr"$\nu / {self.freqs.unit.to_string('latex').strip('$')}$"
        elif xunit == 'loglam':
            x = np.log10(self.wvls.value)
            xx = fr"$\log[\lambda / {self.wvls.unit.to_string('latex').strip('$')}]$"
        elif xunit == 'logfreq':
            x = np.log10(self.freqs.value)
            xx = fr"$\log[\nu / {self.freqs.unit.to_string('latex').strip('$')}]$"
        else:
            raise ValueError(f"{xunit} is not valid")
        if ax is None:
            fig, ax = plt.subplots()
        if flux.ndim == 2:
            for f, n in zip(flux, self.names):
                ax.plot(x, f, label=n, **kwargs)
        else:
            ax.plot(x, flux, label=self.names, **kwargs)
        ax.set_ylabel(f"{fluxtype} / {flux.unit.to_string('latex')}")
        ax.set_xlabel(xx)
        if legend:
            plt.legend()
        return ax


class BasicSpectrum(BasicIndexedSpectra):
    def __init__(self, host_spectra):
        super().__init__(host_spectra, 0)

    def __repr__(self):
        return f"<Spectrum({self.names} / {self.uflam})>"


class HierarchicalMixin:
    def __init__(self, wvls, adu, invadu, calibration, hierarchy):
        self.hierarchy = hierarchy
        names = hierarchy.runids
        groupname = hierarchy.name
        super(HierarchicalMixin, self).__init__(wvls, adu, invadu, calibration, names, groupname)

    @property
    def address(self):
        return self.hierarchy.address

    @classmethod
    def from_hierarchy(cls, hierarchy):
        raise NotImplementedError


class Spectra(HierarchicalMixin, BasicSpectra):
    pass


class Spectrum(HierarchicalMixin, BasicSpectrum):
    pass