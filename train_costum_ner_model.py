import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
from pathlib import Path

#this functionality has to be called via CLI: python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./train.spacy

trainData = [
    ("Venus fragmented, mobile lithosphere may offer a framework for understanding how tectonics on Earth operated in the Archean.", {"entities":[(0, 5, "UAT")]}),
    ("(3) The effects of IMF components on the bow shock position are associated with the draping and pileup of the IMF around the Martian ionosphere; hence, we find that both the subsolar standoff distance and the flaring angle of the bow shock increase with the field strength of the IMF components that are perpendicular to the solar wind flow direction (B<SUB>Y</SUB> and B<SUB>Z</SUB> in the MSO coordinate system).", {"entities":[(325, 335, "UAT")]}),
    ("Better results are obtained when Neptunes eccentricity is excited to e<SUB>N</SUB> ≃ 0.1 and subsequently damped by dynamical friction.", {"entities":[(33, 40, "UAT")]}),
    ("Periodic radar blackouts/weakenings of radar signals at Mars were observed by the Mars Advanced Radar for Subsurface and Ionosphere Sounding/Mars Express, and are associated with these solar energetic electron enhancements.", {"entities":[(56, 60, "UAT")]}),
    ("Lacking a comprehensive theory of chemical evolution capable of explaining the time-dependent pattern of chemical complexification exhibited by our universe, we recommend a bootstrapping approach to mineral classification based on observations of geological field studies, astronomical observations, laboratory experiments, and analyses of natural samples and their environments.", {"entities":[(207, 221, "UAT")]}),
    ("Integral field spectroscopy can map astronomical objects spatially and spectroscopically.", {"entities":[(15, 27, "UAT")]}),
    ("We show that the giant planets in the solar system will experience significant hydrodynamic escape caused by the EUV irradiation from the white dwarf left behind by the Sun.", {"entities":[(38, 50, "UAT")]}),
    ("During this period, the energetic electron fluxes observed near Earth and Mars showed prominent periodic enhancements over four solar rotations, with major periodicities of ∼27 days and ∼13 days.", {"entities":[(74, 78, "UAT")]}),
    ("With GLEAM, users can uniformly process a variety of spectra, including galaxies and active galactic nuclei, in a wide range of instrument setups and signal-to-noise regimes.", {"entities":[(85, 107, "UAT")]}),
    ("The simulation also produces the observed postfall equinox cooling followed by rapid warming in the upper stratosphere.", {"entities":[(106, 118, "UAT")]}),
    ("In particular, we emphasize the importance of performing both of the extinction correction and the direct method of plasma diagnostics simultaneously as an integrated process.",{"entities":[(69, 79, "UAT")]}),
    ("Mars shares many similarities and characteristics to Earth including various geological features and planetary structure.", {"entities":[(0, 4, "UAT")]}),
    ("Current stellar spectroscopic surveys observe hundreds of thousands to millions of stars with (typically) few observational epochs, which allows for binary discovery but makes orbital characterization challenging.", {"entities":[(30, 37, "UAT")]}),
    ("We analyze the range-rate residual data from Cassinis gravity experiment that cannot be explained with a static, zonally symmetric gravity field.", {"entities":[(132, 145, "UAT")]}),
    ("We discuss how the potential acceleration mechanisms and the origins of the FeLoBAL winds may differ for outflows at different locations in quasars.", {"entities":[(140, 147, "UAT")]}),
    ("We leverage neural networks to build a surrogate model that can predict the entire evolution (0-4.5 Gyr) of the 1-D temperature profile of a Mars-like planet for a wide range of values of five different parameters: reference viscosity, activation energy and activation volume of diffusion creep, enrichment factor of heat-producing elements in the crust and initial temperature of the mantle.", {"entities":[(12, 27, "UAT")]}),
    ("The fundamental stellar atmospheric parameters (T<SUB>eff</SUB> and log g) and 13 chemical abundances are derived for medium-resolution spectroscopy from Large Sky Area Multi-Object Fiber Spectroscopic Telescope (LAMOST) Medium Resolution Survey (MRS) data sets with a deep-learning method.", {"entities":[(136, 148, "UAT")]}),
    ("Bulk densities are also calculated according to alternative scenarios along with compositional distributions, and results are compared to solar system objects.", {"entities":[(138, 150, "UAT")]})
]

print("TEST")

nlp = spacy.blank('en') # load a new spacy model
db = DocBin() # create a DocBin object

for text, annot in tqdm(trainData): # data in previous format
    doc = nlp.make_doc(text) # create doc object from text
    ents = []
    for start, end, label in annot['entities']: # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode='contract')
        if span is None:
            print('Skipping entity')
        else:
            ents.append(span)
    try:
        doc.ents = ents # label the text with the ents
        db.add(doc)
    except:
        print(text, annot)

db.to_disk('./train.spacy')