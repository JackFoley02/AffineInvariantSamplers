#!/usr/bin/env python3
#imports
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import rc
from jinja2 import Environment, FileSystemLoader
import subprocess
rc('text', usetex=True)


class SamplerReport:
    """
    Class to facilitate generating report PDF using sampler benchmark test results.
    
    """
    def __init__(self, results, label, transform, overlay = None):
        self.results = results
        self.label = label
        self.transform = transform
        self.overlay = overlay
        self.sampler_names = list(self.results.keys())
        self.compute_stats()

    def generate_variable_labels(self, i1:int, i2:int | None = None, sig:bool = False):
        """
        Generates nice TeX labels if none are provided.

        Arguments:
            i1 : index of 1st parameter
            i2 : index of 2nd parameter
            sig: indicator of whether or not sigma/covariance labels are being generated
        Returns:
            variable : TeX formatted string of the unnamed variable/sigma/covariance
        """
        subscript = str(i1) if i2 is None else str(i1) + ',' + str(i2)

        variable = r'$\sigma_{{{}}}$'.format(subscript) if sig else r'$x_{{{}}}$'.format(subscript)
        return variable

    def compute_stats(self): #collects and computes summary statistics for each sampler
        #Performance stats
        self.tau = {}
        self.ess = {}
        self.elapsed = {}
        self.acceptance = {}
        #Sampler inputs
        self.nsteps = {}
        self.nwalkers = {}
        self.ndims = None
        self.nburn = {}
        #Parameter outputs
        self.note = None
        self.means = {}
        self.sigmas = {}
        self.covs = {}
        #HMC-specific/Dual Averaging Parameters
        self.warmup = {}
        self.epsilon = {}
        self.L = {}
        self.target_accept = {}
        self.gamma = {}
        self.t0 = {}
        self.kappa = {}

        for name, res in self.results.items():
            W, T, D = res['series'].shape #get number of walkers, number of steps, and number of dimensions
            self.nsteps[name] = T #store these parameters for each sampler individually (a bit redundant)
            self.nwalkers[name] = W
            if self.ndims is None:
                self.ndims = D

            self.nburn[name] = res['burn_in'].shape[1] #grab the number of steps ('T') for just the burn-in phase

            self.tau[name] = np.round((res.get('tau', 1.00)), 3)
            self.ess[name] = np.round((res.get('ess', 1.00)), 3)
            self.elapsed[name] = np.round((res.get('time', 0.00)), 3)
            self.acceptance[name] = np.round(np.mean(res.get('acceptance_rates', 0.00)), 3)

            self.epsilon[name] = np.round(res.get('epsilon', 0.00), 3)
            self.target_accept[name] = res.get('target_accept', 0.00)
            self.warmup[name] = res.get('n_warmup', None)
            self.L[name] = res.get('n_leapfrog', None)
            self.gamma[name] = res.get('gamma', None)
            self.t0[name] = res.get('t0', None)
            self.kappa[name] = res.get('kappa', None)
        
            #parameter values, sigmas, and covs.
            sampler_means = {}
            sampler_sigmas = {}
            sampler_covs = {}
            for d in range(D): #loop over all of the dimensions to get means, covariance. 
                parmval = np.round(res['mean'][d], 3)
                parmsig = np.round(np.sqrt(res['cov'][d,d]), 3)
                sampler_means[self.generate_variable_labels(d)] = parmval
                sampler_sigmas[self.generate_variable_labels(d, sig=True)] = parmsig
                if (d % 2) == 1: #if the dimension number is divisible by 2, generate a covariance between the adjacent dimensions
                    parmcov = np.round(res['cov'][d-1, d], 3)
                    sampler_covs[self.generate_variable_labels(d-1, d, sig=True)] = parmcov
                if d == 9: #are there already 10 variables stored? That's probably enough, right?
                    if self.note is None:
                        self.note = f"Only the first 10 parameters are shown. There are {D} total."
                    break
            self.means[name] = sampler_means
            self.sigmas[name] = sampler_sigmas
            self.covs[name] = sampler_covs
            

            #calculating a couple other values for plotting later on:
            self.maxdim = min(self.ndims - 1, 9)
            if self.transform['affine']:
                self.afstring = '_af'
            else:
                self.afstring = ''



    def texdoc(self, template_dir = 'templates'):
        env = Environment(loader = FileSystemLoader(template_dir))

        reportkw = {}
        reportkw['label'] = self.label
        reportkw['D'] = self.ndims

        if self.transform['affine']:
            reportkw['aftitle'] = '(Affine-transformed)'
        else:
            reportkw['aftitle'] = ''

        #Adding images
        #corners and trends first, since there are very many
        #opting to only generate plots for the first 10 parameters, 
        # since beyond that is not informative, and I don't want these reports to be 60 pages long. 
        for num, name in enumerate(self.sampler_names):
            safe_name = name.replace(' ', '_')
            fname_corner = os.path.abspath(f'{self.label}Results/{self.ndims}d{self.afstring}/corner/corner_{safe_name}_dims0-{self.maxdim}.pdf')

            #adding comment if overlays are unavailable:
            overlayc = ''
            if self.overlay is None:
                overlayc = 'Target overlays are unavailable for this distribution.'


            if os.path.exists(fname_corner):
                fig_template = env.get_template('fig_corner.tex')
                reportkw[f'corner_{num}'] = fig_template.render(infile=fname_corner, maxdim = self.maxdim + 1, label = self.label, sampler = name, overlayc = overlayc)
            else:
                print(f'File Not Found: {fname_corner}')
                

            fname_trends = os.path.abspath(f'{self.label}Results/{self.ndims}d{self.afstring}/trends/trends_{safe_name}.pdf')
            if os.path.exists(fname_trends):
                fig_template = env.get_template('fig_trends.tex')
                reportkw[f'trends_{num}'] = fig_template.render(infile = fname_trends, maxdim = self.maxdim + 1, label = self.label, sampler = name)
            else:
                print(f'File Not Found: {fname_trends}')

        #Autocorrelation Function Figure
        fname_autocorr = os.path.abspath(f'{self.label}Results/{self.ndims}d{self.afstring}/autocorrelation_{self.label}.pdf')
        temp_autocorr = env.get_template('fig_autocorr.tex')
        reportkw['fig_autocorr'] = temp_autocorr.render(infile=fname_autocorr, dim = self.ndims, label = self.label)

        #Step Size Tuner
        fname_sst = os.path.abspath(f'{self.label}Results/{self.ndims}d{self.afstring}/StepSize.pdf')
        temp_sst = env.get_template('fig_sst.tex')
        reportkw['fig_sst'] = temp_sst.render(infile=fname_sst, label = self.label, dim = self.ndims)

        #Generating tables
        textable = TexTable(self, env)
        reportkw['tab_summary'] = textable.tab_summary()
        reportkw['tab_params'] = textable.tab_params()
        reportkw['tab_sigmas'] = textable.tab_sigmas()
        reportkw['tab_covs'] = textable.tab_covs()
        reportkw['tab_hmc'] = textable.tab_hmc()

        template = env.get_template('report.tex')
        return template.render(**reportkw)
    
    def write(self, texname = 'report.tex', template_dir = 'templates'):
        tex = self.texdoc(template_dir=template_dir)
        with open(texname, 'w') as f:
            f.write(tex)
        return texname
    
    def compile_pdf(self, texname = None, template_dir = 'templates', latex_compiler = 'pdflatex'):
        if texname is None:
            texname = f"{self.label}_SamplerReport_{self.ndims}{self.afstring}.tex"
        
        self.write(texname=texname, template_dir=template_dir)

        texdir = os.path.dirname(texname) or "."
        texbase = os.path.basename(texname)

        jobname = os.path.splitext(texbase)[0]

        subprocess.run([latex_compiler, f"-jobname={jobname}", texbase], check = True, cwd = texdir)



class TexTable:

    def __init__(self, report, env):
        self.report = report
        self.env = env

    def escape(self, s):
        """
        escapes underscores in a TeX friendly way
        """
        return(str(s).replace('_', r'\_'))
    
    def round_formatting(self, x):
        """
        Formats integers and floats into TeX friendly strings
        """
        if x is None:
            return '--'
        elif isinstance(x, (int, np.integer)):
            return(str(x))
        elif isinstance(x, (float, np.floating)):
            if np.isnan(x):
                return 'NaN'
            else:
                return f"{x:.3f}"
            
        return(self.escape(x))
            
    
    def tab_summary(self):
        headers = ['Sampler Name', 'Acceptance', r'$\tau$', 'ESS', 'Time (s)', 'Walkers', 'Steps', 'Burn-in Steps']

        rows = []
        for name in self.report.sampler_names:
            rows.append([
                self.escape(name),
                self.round_formatting(self.report.acceptance[name]),
                self.round_formatting(self.report.tau[name]),
                self.round_formatting(self.report.ess[name]),
                self.round_formatting(self.report.elapsed[name]),
                self.round_formatting(self.report.nwalkers[name]),
                self.round_formatting(self.report.nsteps[name]),
                self.round_formatting(self.report.nburn[name])
            ])

        template = self.env.get_template('tab_summary.tex')

        return template.render(title = f'Sampler Performance Summary for {self.report.label} Distribution, {self.report.ndims} Dimensions', headers = headers, rows=rows)
    
    def tab_params(self):
        """
        Generates TeX code for a table that displays the first 10 parameter values. 
        """
        fk = self.report.sampler_names[0]
        xheaders = list(self.report.means[fk].keys())
        headers = ['Name'] + xheaders
        colspec = 'l' + ('r'*(len(headers) -1 ))

        if self.report.note is not None:
            note = self.report.note
        else:
            note = 'All parameters displayed.'

        rows = []
        for name in self.report.sampler_names:
            row = [self.escape(name)]
            for label in xheaders:
                row.append(self.round_formatting(self.report.means[name].get(label)))
            rows.append(row)

        template = self.env.get_template('tab_params.tex')

        return template.render(title = f'Parameters for {self.report.label} Distribution, {self.report.ndims} Dimensions', headers = headers, rows = rows, colspec = colspec, note = note)

    def tab_sigmas(self):
        """
        Generates TeX code for a table that displays the first 10 parameter sigmas. 
        """
        fk = self.report.sampler_names[0]
        sigheaders = list(self.report.sigmas[fk].keys())
        headers = ['Name'] + sigheaders
        colspec = 'l' + ('r'*(len(headers) - 1))

        if self.report.note is not None:
            note = self.report.note
        else:
            note = 'All parameters displayed.'

        rows = []
        for name in self.report.sampler_names:
            row = [self.escape(name)]
            for label in sigheaders:
                row.append(self.round_formatting(self.report.sigmas[name].get(label)))
            rows.append(row)

        template = self.env.get_template('tab_params.tex')

        return template.render(title = f'Parameter Standard Deviations for {self.report.label} Distribution, {self.report.ndims} Dimensions', headers = headers, rows = rows, colspec = colspec, note = note)

    
    def tab_covs(self):
        """
        Generates TeX code for a table that displays the covariances of each adjacent parameter
        """
        fk = self.report.sampler_names[0]
        covheaders = list(self.report.covs[fk].keys())
        headers = ['Name'] + covheaders
        colspec = 'l' + ('r'*(len(headers)-1))

        if self.report.note is not None:
            note = self.report.note
        else:
            note = 'All parameters displayed'
        rows = []
        for name in self.report.sampler_names:
            row = [self.escape(name)]
            for label in covheaders:
                row.append(self.round_formatting(self.report.covs[name].get(label)))
            rows.append(row)

        template = self.env.get_template('tab_params.tex')

        return template.render(title = f'Parameter Covariances for {self.report.label} Distribution, {self.report.ndims} Dimensions', headers = headers, rows = rows, colspec = colspec, note = note)
        
    def tab_hmc(self):
        """
        Generates a table that lists all of the HMC tuning parameters
        """
        headers = [
        'Name',
        r'\ensuremath{\epsilon}',
        'L',
        'Warm-Up (steps)',
        'Target Acceptance',
        r'\ensuremath{\gamma}',
        't0',
        r'\ensuremath{\kappa}'
        ]    
        rows = []
        for name in self.report.sampler_names:
            if self.report.warmup[name] is None:
                continue
            rows.append([
                self.escape(name),
                self.round_formatting(self.report.epsilon[name]),
                self.round_formatting(self.report.L[name]),
                self.round_formatting(self.report.warmup[name]),
                self.round_formatting(self.report.target_accept[name]),
                self.round_formatting(self.report.gamma[name]),
                self.round_formatting(self.report.t0[name]),
                self.round_formatting(self.report.kappa[name])
            ])

        template = self.env.get_template('tab_hmc.tex')
        return template.render(title = f'HMC Tuning Parameters for {self.report.label} Distribution, {self.report.ndims} Dimensions', headers = headers, rows = rows)

        
