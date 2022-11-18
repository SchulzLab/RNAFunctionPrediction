
library(DESeq2)
library(data.table)

# Read in the human genes
#humanCounts <- read.table(file = 'GeneExpressionDiffCellTypes/counts_gene.tsv', sep = '\t', header = TRUE)

#humanCounts <- read.table(file="GeneExpressionDiffCellTypes/counts_gene.tsv", sep = '\t', colClasses=c("geneid"), header = TRUE, nrows=2)
print("Start reading in the expression file")
humanCounts <- read.table(file = "GeneExpressionDiffCellTypes/counts_gene.tsv", sep='\t', header=TRUE)

print( paste("Number of found gene examples: ", length(humanCounts$gene_id))) 
# Extract the ensembl ids for retrieve
geneIds <- humanCounts$gene_id

# Discard the version of the enseml ID
print("Discarding the Ensembl Id Version")
genesCut <- gsub("\\..*","",c(geneIds))

# Discard the gene_id for normalization
dim(humanCounts)
humanCounts <- humanCounts[ , -which(names(humanCounts) %in% c("gene_id"))]
dim(humanCounts)
humanCounts[is.na(humanCounts)] <- 0

max(humanCounts)
max(round(humanCounts/2))
.Machine$integer.max
#test <- colnames(data.table::fread( 
#  "GeneExpressionDiffCellTypes/counts_gene.tsv"  ,  
#  drop = "gene_id"
#))
print("Construct the DESeqMatrix")
#data <- humanCounts[ , -which(names(humanCounts) %in% c("gene_id"))]



colData <- data.frame(row.names=colnames(humanCounts))
#counts <- as.matrix(humanCounts)

### Check that sample names match in both files
all(colnames(humanCounts) %in% rownames(colData))
all(colnames(humanCounts) == rownames(colData))
ncol(humanCounts)
nrow(colData)

dds <- DESeqDataSetFromMatrix(
  countData=round(humanCounts/8),
  colData=colData,
  design = ~ 1
)


# Normalize
print("Start normalization")
dds <- estimateSizeFactors(dds)
humanCounts <- counts(dds,normalized=TRUE)
dds <- ''
#humanCounts <- as.data.frame(humanCounts)
#humanCounts$gene_id <- geneIds$gene_id
print("Normalization finished and export initiated")

fwrite(list(geneIds), file = "homoSapiensGeneIds.csv")
write.csv(humanCounts,"Datasets/homoSapiensExpressionsTab.csv", sep = "\t", row.names = TRUE)
