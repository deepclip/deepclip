## DeepCLIP

Welcome to DeepCLIP's git repository. 

DeepCLIP is a novel deep learning tool for finding binding-preferences of RNA-binding proteins. Use pre-trained models or train your own at [http://deepclip.compbio.sdu.dk/](http://deepclip.compbio.sdu.dk).

In this repository, you can find all relevant code for running DeepCLIP on your local machine.

Summary of DeepCLIP and its functionalities:
* DeepCLIP is a neural network with shallow convolutional layers connected to a bidirectional LSTM layer.
* DeepCLIP can calculate binding profiles and pseudo position frequency matrices.
* Binding profiles show whether areas of sequences contain possible binding sites or whether they look like random genomic background.
* DeepCLIP outperforms current state-of-the-art RNA-binding protein motif discovery tools on curated CLIP datasets.
---
## Table of contents
* [Requirements](#requirements)
* [Installation](#installation)
* [Usage](#usage)
* [Models](#Models)
* [Citation](#citation)
* [Contributors](#contributors)
---
## Requirements
DeepCLIP was designed to run on Linux flavoured operating systems and while it may run on Windows or FreeBSD flavours such as OS-X we do not actively support this.

DeepCLIP requires Python 2.7 along with the latest versions of Theano and Lasagne.
To install requirements for DeepCLIP, please install Theano and then Lasagne, followed by the remaining requirements:
```shell
pip install git+git://github.com/Theano/Theano.git
pip install https://github.com/Lasagne/Lasagne/archive/master.zip
pip install mkl-service
pip install scikit-learn
pip install matplotlib
pip install biopython
pip install htseq
```

We recommend using conda to install a DeepCLIP specific environment along with DeepCLIP requirements:
```shell
conda create -n deepclip python=2.7 mkl-service numpy scipy scikit-learn biopython htseq matplotlib
conda activate deepclip
pip install git+git://github.com/Theano/Theano.git
pip install https://github.com/Lasagne/Lasagne/archive/master.zip
```
---
## Installation

DeepCLIP can be run from within its source directory, so to install DeepCLIP simply clone this repository and add it to your path environment variable.
```shell
git clone http://github.com/deepclip/deepclip
```

In order to run DeepCLIP, simply run DeepCLIP.py:
```shell
$install_folder/DeepCLIP.py
```
## Usage

DeepCLIP can be used to train either directly on binding site locations in BED format, or on pre-made positive and negative classes in FASTA format. To train a model from binding sites in BED format, we recommend using a matched genomic background to generate a set of background sequences based on gene annotations given to DeepCLIP in GTF format. DeepCLIP also requires the genome sequence in FASTA format.

DeepCLIP saves model data in a pickled format via the '-n' and '-P' options. The first option saves the weights of the trained neural network and can be transfered between computers. The second option saves a pre-compiled prediction function that is generally only applicable on the same computer that it was trained on. DeepCLIP can use either as input when predicting on new sequences, but it is time-saving to use the pre-compiled prediction function as this can be loaded in a few seconds.

### Compiling prediction function from saved network weights
If you have only saved the network weights via the '-n' option, you can use the 'recompile.py' file in the DeepCLIP folder to compile the prediction function from the network weights:
```shell
python $install_folder/recompile.py MODEL MODEL_PREDICTION_FUNCTION
```


### Training with binding sites alone in BED format
To train a model using binding sites in BED format and gene annotations in GTF format, execute the following:
```shell
DeepCLIP.py --runmode train -n MODEL -P MODEL_PREDICTION_FUNCTION --sequences INPUT.BED --gtf_file GENES.GTF --genome_file GENOME.FA
```
Note that DeepCLIP tries to detect the input format of the sequences by the file suffix, if you have BED formatted data in a file without the .bed suffix, you can force BED input format with the '--force_bed' option. Otherwise, DeepCLIP will assume FASTA format as the default input format.

If you wish to construct a set of binding sites and background sequences to use over and over, you can use the 'bed.py' file to produce the FASTA files directly and save for later use.
```shell
python $install_folder/bed.py INPUT.BED GENOME.FA GENES.GTF pos.fa neg.fa
```

### Training with binding sites and background in FASTA format
To train a model using both binding sites and background in FASTA format, execute the following:
```shell
DeepCLIP.py --runmode train -n MODEL -P MODEL_PREDICTION_FUNCTION --sequences POSITIVES.FASTA --background_sequences NEGATIVES.FASTA
```

### Adjusting training parameters
DeepCLIP allows adjustment of a number of training parameters, first and foremost the number of maximum training epochs used. Here an 'epoch' refers to one full processing of the training set.

To adjust the number of training epochs, use the '-e' or '--num_epochs' options followed by an integer, e.g:
```shell
DeepCLIP.py --runmode train -e 100 -n MODEL -P MODEL_PREDICTION_FUNCTION --sequences POSITIVES.FASTA --background_sequences NEGATIVES.FASTA
```

DeepCLIP supports early stopping, meaning that if no improvement of the model has been obtained for N epochs in a row, training is stopped before the maximum number of epochs is reached in order to reduce computation time.
To adjust the number of epochs in a row without model improvement, use the '--early_stopping' followed by an integer, e.g.:
```shell
DeepCLIP.py --runmode train -e 100 --early_stopping 10 -n MODEL -P MODEL_PREDICTION_FUNCTION --sequences POSITIVES.FASTA --background_sequences NEGATIVES.FASTA
```

DeepCLIP automatically partitions the input data into sets of training data, validation data and final test data. Training is performed on the training set and after each epoch the performance is measured on the validation set. After training is completed the best performing model based on the validation set can be tested on the test data, which was not used at any point during model training.
By default, DeepCLIP partitions the input data such that 80% is used for training, 10% for validation, and 10% for testing. If you wish to change this behaviour you can do so with the '--data_split' option followed by 3 floats corresponding to training, validation, test partitions. The floats must sum to 1, e.g:
```shell
DeepCLIP.py --runmode train --data_split 0.75 0.15 0.1 -n MODEL -P MODEL_PREDICTION_FUNCTION --sequences POSITIVES.FASTA --background_sequences NEGATIVES.FASTA
```
---
### Producing visualizations of CNN filters
DeepCLIP allows generation of CNN filters via the '--predict_PFM_file' option during training. This exports pseudo-PFM data in JSON format, which can be plotted using the R packages ggplot2 and ggseqlogo.
To enable saving of CNN filter data during training, set the '--predict_PFM_file' option followed by the file name you wish to save the data to: 
```shell
DeepCLIP.py --runmode train --predict_PFM_file pfms.json -n MODEL -P MODEL_PREDICTION_FUNCTION --sequences POSITIVES.FASTA --background_sequences NEGATIVES.FASTA
```
The "pfms.json" file can then be processed in R:
```r
library(jsonlite)
library(ggplot2)
library(ggseqlogo)

data <- jsonlite::read_json("pfms.json")
scores <- sapply(data[["logos"]], function(logo) {
  pfm <- do.call(rbind, lapply(logo[["pfm"]], unlist))
  sum(colSums(pfm * log2(pfm + 0.000000001)) + log2(4)) / ncol(pfm)
})
seq_letters <- c("A","C","G","U")
logos <- lapply(data[["logos"]], function(logo) {
  weights <- lapply(logo[["pfm"]], unlist)
  weights <- do.call(rbind, weights)
  rownames(weights) <- seq_letters
  weights
})
names(logos) <- formatC(scores)
logos <- logos[order(scores, decreasing=TRUE)]

# all ranked in the same plot
p <- ggplot() +
  geom_logo(logos, method="probability") +
  theme_logo() +
  facet_grid(seq_group ~ ., switch = "y") +
  ggtitle("CNN filters ranked by score") +
  theme(
    axis.title = element_blank(),
    strip.text.y = element_text(angle = 180, size=16),
    panel.spacing.y = unit(5, "pt"),
    axis.text.y = element_blank())

pdf("pfms.pdf")
print(p)
dev.off()
```
---
### Running 10-fold cross-validation
In order to obtain the best model from a given set of binding sites, DeepCLIP can be run in 10-fold cross validation mode (CV mode). Int his mode, DeepCLIP creates 10 different partitions of the data by dividing it into 10 bins. Each bin is used exactly once as a test set, once as a validation set, and 8 times as a training set. In this mode, the '--data_split' option is ignored. After the 10 different models are trained the best performing one is used to compile the final prediction function.
To train 10 models in CV mode, execute the following:
```shell
DeepCLIP.py --runmode cv -n MODEL -P MODEL_PREDICTION_FUNCTION --predict_PFM_file pfms.json --sequences POSITIVES.FASTA --background_sequences NEGATIVES.FASTA
```
This results in the generation of a number of files, so we suggest running CV in its own folder.
The files are:
* *MODEL*_cv[1-10]-predictions_id_seq_pred.txt 

Prediction results from each of the 10 different test sets.

* *MODEL*_cv[1-10]-predictions.txt 

Two column files with the input class (0 or 1) in the first column and the prediction value in the second. One for each CV cycle.

* *MODEL*_cv[1-10]

Saved weights from each model.

* *MODEL*_best_cv_model

Saved weights from the best model.

* *MODEL*_best_cv_predict_fn

Saved prediction function from the best model.

* *MODEL*_cv_cycle_data.pkl

Saved CV data for resuming an aborted CV run.

### Resuming an aborted 10-fold cross-validation run
Running DeepCLIP in CV mode can be time-consuming depending on the size of the training set and the number of epochs used during training. This may result in the training process being aborted on some systems, such as HPC clusters with time limits on nodes, e.g. not being able to run a node continuously for more than 24 hours.

In order to allow DeepCLIP to resume an aborted CV runmode, data about the CV run is saved after each cycle to allow DeepCLIP to resume training of models for the remainder of CV cycles. In this case, it is important that the random seed number given is the same as in the original run, otherwise DeepCLIP will partition the data differently and the CV run has to be restarted completely.

To resume an aborted CV run, use the *MODEL*_cv_cycle_data.pkl as network name when starting a CV run:
```shell
DeepCLIP.py --runmode cv -n MODEL_cv_cycle_data.pkl -P MODEL_PREDICTION_FUNCTION --predict_PFM_file pfms.json --sequences POSITIVES.FASTA --background_sequences NEGATIVES.FASTA
```

---
### Predicting on sequences in single mode.
DeepCLIP reads sequence input in FASTA format.

To use a model for prediction and save the results as a TSV file, execute the following:
```shell
DeepCLIP.py --runmode predict -P MODEL_PREDICTION_FUNCTION --sequences INPUT.FASTA --predict_output_file RESULTS.tsv
```

If you want to save the results in JSON format for further processing, please use the .json suffix:
```shell
DeepCLIP.py --runmode predict -P MODEL_PREDICTION_FUNCTION --sequences INPUT.FASTA --predict_output_file RESULTS.json
```

The JSON file will contain the input names, sequences, overall prediction scores, and per-nucleotide binding profile data.

### Predicting on sequences in paired mode.
DeepCLIP supports variant analysis by running prediction in paired mode. In this mode, DeepCLIP considers input sequences given as pairs using the '--sequences' and '--variant_sequences' options. The input order must be kept such that input sequence #31 has a corresponding variant sequence as variant sequence #31. The only exception is when all variant sequences are variants of a single reference sequence. In this mode, only a single input sequence should be given using the '--sequences' option.

To use a model for prediction of sequences in paired mode and save the results as a TSV file, execute the following:
```shell
DeepCLIP.py --runmode predict -P MODEL_PREDICTION_FUNCTION --sequences REFERENCES.FASTA variant_sequences VARIANTS.FASTA --predict_output_file RESULTS.tsv
```

If you want to save the results in JSON format for further processing, please use the .json suffix:
```shell
DeepCLIP.py --runmode predict -P MODEL_PREDICTION_FUNCTION --sequences REFERENCES.FASTA variant_sequences VARIANTS.FASTA --predict_output_file RESULTS.json
```

---
### Plotting binding profiles
DeepCLIP supports plotting of binding profiles using *matplotlib* directly via the '--draw-profiles' option when running in prediction mode:
```shell
DeepCLIP.py --runmode predict -P MODEL_PREDICTION_FUNCTION --sequences REFERENCES.FASTA variant_sequences VARIANTS.FASTA --predict_output_file RESULTS.json --draw_profiles
```

The binding profiles are saved as PNG images with names based on the sequence names in the input. It is therefor important to take care in not having identically named sequences in the input files.

In single prediction mode the binding profile is drawn in black, and in paired mode the variant binding profile is added as a red line to the plot and the variant sequence differences indicated in red below the reference sequence.

In paired mode, the difference between them can be plotted by adding the '--make_diff' option:
```shell
DeepCLIP.py --runmode predict -P MODEL_PREDICTION_FUNCTION --sequences REFERENCES.FASTA variant_sequences VARIANTS.FASTA --predict_output_file RESULTS.json --draw_profiles --make_diff
```
In this mode, the reference profile is plotted along with the difference between the variant and the reference profile in red.

From the JSON output, binding profiles can also be generated from both single prediction mode and paired prediction mode. We suggest using R to plot the data:
```r
library(jsonlite)
library(ggplot2)
library(ggpubr)

mytheme <- function() {
  ggpubr::theme_pubclean() +
    theme(
      axis.line = element_line(color="black"),
      axis.text = element_text(color="black"),
      strip.text = element_text(face="bold")
    )
}

# single prediction mode
data <- jsonlite::fromJSON("predictions.single.json")$predictions

# make sure it's RNA sequence
data$sequence = gsub("T", "U", data$sequence)
data$sequence = gsub("t", "u", data$sequence)

for (i in 1:length(data$sequence)) {
  tbl <- data.frame(
    pos = 1:length(data$weights[[i]]),
    weight = data$weights[[i]],
    group = factor(rep("Profile", length(data$weights[[i]]))))
  xlabels <- strsplit(data$sequence[i], "")[[1]]
  p <- ggplot(tbl, aes(pos, weight))
  p <- p +
    geom_line(aes(color=group), size=0.8) +
    scale_x_continuous(breaks=seq(1, max(tbl$pos)), labels=xlabels) +
    theme_bw() +
    theme(
      legend.title = element_blank(),
      axis.title.x = element_blank(),
      axis.text.x = element_text(size=6)
    ) + labs(y="DeepCLIP score")
  pdf(paste0("binding_profile.", gsub(":", "_", data$id[i]),".pdf"), height = 6, width = as.integer(length(xlabels)/8))
  print(p)
  dev.off()
}

# paired prediction mode
data <- jsonlite::fromJSON("predictions.paired.json")$predictions

# make sure it's RNA sequence
data$sequence = gsub("T", "U", data$sequence)
data$sequence = gsub("t", "u", data$sequence)
data$variant_sequence = gsub("T", "U", data$variant_sequence)
data$variant_sequence = gsub("t", "u", data$variant_sequence)

make_paired_profile_plot <- function(x, plot_difference) {
  weights1 <- unlist(x$weights)
  weights2 <- unlist(x$variant_weights)
  
  seq1 <- strsplit(toupper(x$sequence), "")[[1]]
  seq2 <- strsplit(toupper(x$variant_sequence), "")[[1]]
  
  if(plot_difference) {
    weights2 <- weights2 - weights1
    tbl <- data.frame(
      pos = seq_along(seq2),
      weight = weights2,
      group = factor(rep("difference", length(seq2)))
    )
  } else {
    tbl <- data.frame(
      pos = c(seq_along(seq1), seq_along(seq2)),
      weight = c(weights1, weights2),
      group = factor(c(rep("reference", length(seq1)), rep("variant", length(seq2))), levels=c("reference","variant"))
    )
  }
  
  xlabels <- mapply(function(a, b) paste(a, ifelse(a==b, "", b), sep="\n"), seq1, seq2)
  
  p <- ggplot(tbl, aes(pos, weight))
  if(plot_difference) p <- p + geom_hline(yintercept=0, color="dodgerblue")
  p <- p +
    geom_line(aes(color=group), size=0.8) +
    scale_x_continuous(breaks=seq(1, max(tbl$pos)), labels=xlabels) +
    scale_color_manual(values=c("black", "red")) +
    mytheme() +
    theme(
      legend.title = element_blank(),
      axis.title.x = element_blank(),
      axis.text.x = element_text(size=11)
    ) + labs(y="DeepCLIP score")
  return(p)
}

for (i in 1:dim(data)[1]) {
  x = data[i,]
  width = 10.5
  height = 3.65
  if (length(x$weights[[1]]) <= 30) {width = 7.75}
  p = make_paired_profile_plot(x, plot_difference = FALSE)
  pdf(paste0("profile_",i,".pdf"), width = width, height = height)
  print(p)
  dev.off()

  p_diff = make_paired_profile_plot(x, plot_difference = TRUE)
  pdf(paste0("profile_",i,".difference.pdf"), width = width, height = height)
  print(p_diff)
  dev.off()
}

```
---
### Predicting long sequences
Sequences longer than the model was trained on can be predicted on in long prediction mode. In this mode, the binding profile is created by predicting on smaller segments of the sequence and employing a sliding window algorithm to build the binding profile. 
In this mode there is no overall prediction score and no paired prediction.
To run DeepCLIP in long prediction mode, use "predict_long" as runmode:
```shell
DeepCLIP.py --runmode predict_long -P MODEL_PREDICTION_FUNCTION --sequences LONG.FASTA --predict_output_file predictions.long.json
```
Binding profiles can then be extracted and analyzed in R, e.g.:
```r
library(jsonlite)
library(ggplot2)
library(ggpubr)

mytheme <- function() {
  ggpubr::theme_pubclean() +
    theme(
      axis.line = element_line(color="black"),
      axis.text = element_text(color="black"),
      strip.text = element_text(face="bold")
    )
}

data <- jsonlite::fromJSON("predictions.long.json")$predictions

# make sure it's RNA sequence
data$sequence = gsub("T", "U", data$sequence)
data$sequence = gsub("t", "u", data$sequence)

for (i in 1:length(data$sequence)) {
  tbl <- data.frame(
    pos = 1:length(data$weights[[i]]),
    weight = data$weights[[i]],
    group = factor(rep("Profile", length(data$weights[[i]]))))
  xlabels <- strsplit(data$sequence[i], "")[[1]]
  p <- ggplot(tbl, aes(pos, weight))
  p <- p + geom_hline(yintercept=0, color="dodgerblue")
  p <- p +
    geom_line(size=0.8) +
    scale_x_continuous(breaks=seq(1, max(tbl$pos)), labels=xlabels) +
    mytheme() +
    theme(
      legend.title = element_blank(),
      axis.title.x = element_blank(),
      axis.text.x = element_text(size=6)
    ) + labs(y="DeepCLIP score")
  pdf(paste0("long_profile.", gsub(":", "_", data$id[i]),".pdf"), height = 6, width = as.integer(length(xlabels)/8))
  print(p)
  dev.off()
}

```
---
## Models
In [deepclip/models](https://github.com/deepclip/models) you can find all models used in the article presenting DeepCLIP (see below).

---
## Citation
The DeepCLIP preprint is now available at bioRxiv: <https://doi.org/10.1101/757062>

---
## Contributors
Main DeepCLIP development
* Alexander Gulliver Bjørnholt Grønning
* Thomas Koed Doktor

Additional code
* Simon Jonas Larsen
