# OVERVIEW
# Bivariate & Univariate Analysis
# Data Cleaning
# Data Preprocessing & Sampling
# Unsupervised & Supervised Machine Learning
# Segmentation of Customers
# Hyperparameter Tuning
# Predictive Modelling with XGBoost to classify the Risk.
# ROC Analysis
install.packages("gridExtra", repos = "https://cran.rstudio.com/")

install.packages(c("ggplot2", "ggdist", "gghalves", "patchwork", "RColorBrewer",
                   "dplyr", "tidyverse", "data.table", 
                   "ggthemes", "plotly", "caret", "randomForest",
                   "xgboost", "class", "cluster", "stats",
                   "BayesianTools", "gridExtra", "patchwork", "viridis",
                   "RColorBrewer", "psych", "grid", "GGally", "purrr", "tidyr",
                   "gghalves","ggdist", "ggprism", "stats", "scales", "reshape2
                   ", "pheatmap", "cluster", "rgl"))

library(cluster)
library(pheatmap) 
library(reshape2)
library(rgl)
library(scales) 
library(stats)
library(ggplot2)
library(ggdist)       
library(gghalves)    
library(patchwork)    
library(RColorBrewer)
library(dplyr)
library(plotly)
library(caret)
library(randomForest)
library(xgboost)
library(cluster)
library(stats)
library(BayesianTools)
library(class)
library(ggthemes)
library(data.table)
library(tidyverse)
library(gridExtra)
library(viridis)
library(psych)
library(grid)
library(GGally)
library(ggforce)
library(tidyr)
library(gghalves)
library(ggdist)
library(purrr)


options(warn = -1)
# Charger le fichier CSV dans un dataframe
library(readr)
df <- read_csv("german_credit_data.csv")
# Afficher les premières lignes du dataframe
str(df)
head(df)


show_info <- function(data) {
  # Afficher la taille du dataset
  cat("DATASET SHAPE:", dim(data), "\n")
  cat(rep("-", 50), "\n")
  
  # Afficher les types de données
  cat("FEATURE DATA TYPES:\n")
  print(str(data))
  cat("\n", rep("-", 50), "\n")
  
  # Nombre de valeurs uniques par colonne
  cat("NUMBER OF UNIQUE VALUES PER FEATURE:\n")
  print(sapply(data, function(x) length(unique(x))))
  cat("\n", rep("-", 50), "\n")
  
  # Nombre de valeurs manquantes par colonne
  cat("NULL VALUES PER FEATURE:\n")
  print(colSums(is.na(data)))
}


##EDA
show_info(df)

#UNIVARIATE ANALYSIS

  # Pour organiser les graphiques

#Voici la version RStudio de votre code Python, utilisant ggplot2 pour la visualisation :
  
  #📊 Distribution Plots (Histogrammes)

# Installer et charger les packages nécessaires



# Histogrammes avec densité (distribution plots)
p1 <- ggplot(df, aes(x = `Credit amount`)) +
  geom_histogram(aes(y = ..density..), bins = 40, fill = "steelblue", alpha = 0.6) +
  geom_density(color = "steelblue", size = 1.2) +
  theme_minimal()

p2 <- ggplot(df, aes(x = Duration)) +
  geom_histogram(aes(y = ..density..), bins = 40, fill = "salmon", alpha = 0.6) +
  geom_density(color = "salmon", size = 1.2) +
  theme_minimal()

p3 <- ggplot(df, aes(x = Age)) +
  geom_histogram(aes(y = ..density..), bins = 40, fill = "darkviolet", alpha = 0.6) +
  geom_density(color = "darkviolet", size = 1.2) +
  theme_minimal()

# Affichage côte à côte avec titre global
(p1 | p2 | p3) + 
  plot_annotation(title = "DISTRIBUTION PLOTS",
                  theme = theme(plot.title = element_text(hjust = 0.5, size = 16)))


# BOX PLOTS horizontaux sans titres d'axe y


df$var <- "1"

# Boxplot 1 : Credit amount
b1 <- ggplot(df, aes(x = var, y = `Credit amount`)) +
  geom_boxplot(fill = "steelblue") +
  theme_minimal() +
  labs(x = "Credit amount", y = NULL) +
  theme(
    axis.title.x = element_text(hjust = 0.5, vjust = -1),  # Centrer le titre
    axis.text.x = element_blank(),       # Masquer les labels de l'axe X
    axis.ticks.x = element_blank()       # Masquer les ticks de l'axe X
  ) +
  coord_flip()  # Faire pointer les moustaches vers la droite

# Boxplot 2 : Duration
b2 <- ggplot(df, aes(x = var, y = Duration)) +
  geom_boxplot(fill = "salmon") +
  theme_minimal() +
  labs(x = "Duration", y = NULL) +
  theme(
    axis.title.x = element_text(hjust = 0.5, vjust = -1),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  coord_flip()

# Boxplot 3 : Age
b3 <- ggplot(df, aes(x = var, y = Age)) +
  geom_boxplot(fill = "darkviolet") +
  theme_minimal() +
  labs(x = "Age", y = NULL) +
  theme(
    axis.title.x = element_text(hjust = 0.5, vjust = -1),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  coord_flip()

# Affichage des boxplots côte à côte avec un titre général
grid.arrange(b1, b2, b3, ncol = 3, top = "BOX PLOTS")



#INSIGHTS

#Most of the credit cards have an amount of 1500 - 4000
#The Credit amount is positively skewed, So the samples are dispersed

#COUNTPLOTS (SEX & RISK FACTOR)
 # Pour afficher plusieurs graphiques côte à côte

# Chargement des bibliothèques nécessaires




# Réordonner les facteurs selon le nouvel ordre souhaité
df$Sex <- factor(df$Sex, levels = c("male", "female"))  # Male à gauche, Female à droite
df$Risk <- factor(df$Risk, levels = c("good", "bad"))   # Bad à gauche, Good à droite (inchangé)

# Couleurs personnalisées
sex_colors <- c("male" = "#8E24AA",    # mauve
                "female" = "#C2185B")  # rose vin

risk_colors <- c("bad"  = "#EF6C00",   # orange foncé
                 "good" = "#F06292")   # rose clair

# Barplot Sex
p1 <- ggplot(df, aes(x = Sex, fill = Sex)) +
  geom_bar() +
  scale_fill_manual(values = sex_colors) +
  theme_minimal() +
  labs(title = "Count Plot: Sex", x = "Sex", y = "Count") +
  theme(legend.position = "none")

# Barplot Risk
p2 <- ggplot(df, aes(x = Risk, fill = Risk)) +
  geom_bar() +
  scale_fill_manual(values = risk_colors) +
  theme_minimal() +
  labs(title = "Count Plot: Risk", x = "Risk", y = "Count") +
  theme(legend.position = "none")

# Affichage côte à côte
grid.arrange(p1, p2, ncol = 2)



describe(df[, c("Age", "Duration", "Credit amount")])



#BIVARIATE ANALYSIS



main_title <- "BIVARIATE ANALYSIS (HUE=SEX)"

p1 <- ggplot(df, aes(x = Age, y = `Credit amount`, color = Sex)) +
  geom_line(size = 1.5) +
  geom_smooth(method = "loess", se = TRUE, colour = NA, fill = "grey80", alpha = 0.4, span = 1) +  # Nuage large uniquement
  theme_minimal() +
  ggtitle("Age vs Credit amount") +
  theme(plot.title = element_text(hjust = 0.5))

p2 <- ggplot(df, aes(x = Duration, y = `Credit amount`, color = Sex)) +
  geom_line(size = 1.5) +
  geom_smooth(method = "loess", se = TRUE, colour = NA, fill = "grey80", alpha = 0.4, span = 1) +  # Nuage large uniquement
  theme_minimal() +
  ggtitle("Duration vs Credit amount") +
  theme(plot.title = element_text(hjust = 0.5))

p3 <- ggplot(df, aes(x = Age, y = Duration, color = Sex)) +
  geom_line(size = 1.5) +
  geom_smooth(method = "loess", se = TRUE, colour = NA, fill = "grey80", alpha = 0.4, span = 1) +  # Nuage large uniquement
  theme_minimal() +
  ggtitle("Age vs Duration") +
  theme(plot.title = element_text(hjust = 0.5))

grid.arrange(
  p1, p2, p3,
  ncol = 1,
  top = textGrob(main_title, gp = gpar(fontsize = 20, fontface = "bold")))



main_title <- "BIVARIATE ANALYSIS (HUE=RISK)"

palette_deep <- brewer.pal(n = length(unique(df$Risk)), name = "Dark2")

p1 <- ggplot(df, aes(x = Age, y = `Credit amount`, color = Risk)) +
  geom_line(size = 1.5) +
  geom_smooth(method = "loess", se = TRUE, colour = NA, fill = "grey80", alpha = 0.3) +  # Nuage uniquement
  scale_color_manual(values = palette_deep) +
  theme_minimal() +
  ggtitle("Age vs Credit amount") +
  theme(plot.title = element_text(hjust = 0.5))

p2 <- ggplot(df, aes(x = Duration, y = `Credit amount`, color = Risk)) +
  geom_line(size = 1.5) +
  geom_smooth(method = "loess", se = TRUE, colour = NA, fill = "grey80", alpha = 0.3) +  # Nuage uniquement
  scale_color_manual(values = palette_deep) +
  theme_minimal() +
  ggtitle("Duration vs Credit amount") +
  theme(plot.title = element_text(hjust = 0.5))

p3 <- ggplot(df, aes(x = Age, y = Duration, color = Risk)) +
  geom_line(size = 1.5) +
  geom_smooth(method = "loess", se = TRUE, colour = NA, fill = "grey80", alpha = 0.3) +  # Nuage uniquement
  scale_color_manual(values = palette_deep) +
  theme_minimal() +
  ggtitle("Age vs Duration") +
  theme(plot.title = element_text(hjust = 0.5))

grid.arrange(
  p1, p2, p3,
  ncol = 1,
  top = textGrob(main_title, gp = gpar(fontsize = 20, fontface = "bold")))



#INSIGHTS
#There is a linear relationship between Duration and Creadit Amount, Which makes sense because usually, people take bigger credits for longer periods.
#The trend Between Age and Credit amount is not clear.


#PAIRPLOT TO VISUALIZE FEATURES WITH LINEAR RELATIONSHIP

# Installer le package GGally si ce n'est pas déjà fait



# Select only the columns of interest
df_subset <- df[, c("Job", "Duration", "Credit amount", "Age")]

ggpairs(df_subset,
        diag = list(continuous = "barDiag"),   # histograms on diagonal
        upper = list(continuous = "points"),   # scatter plots on upper triangle
        lower = list(continuous = "points")    # scatter plots on lower triangle
) +
  ggtitle("Histograms on Diagonal and Scatterplots Off-Diagonal for Selected Variables") +
  theme_minimal()





#SAVING ACCOUNT ANALYSIS

# Installer ggforce si nécessaire


# Graphique 1 : Count plot 'Saving accounts' par 'Risk'
p1 <- ggplot(df, aes(x = `Saving accounts`, fill = Risk)) +
  geom_bar(position = "dodge") +
  scale_fill_brewer(palette = "Greens") +
  ggtitle("Count Plot: Saving accounts by Risk") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Graphique 2 : Boxenplot 'Credit amount' par 'Saving accounts' et 'Risk'
p2 <- ggplot(df, aes(x = `Saving accounts`, y = `Credit amount`, fill = Risk)) +
  geom_boxen(position = position_dodge(width = 0.8)) +
  scale_fill_brewer(palette = "Greens") +
  ggtitle("Boxenplot: Credit amount by Saving accounts and Risk") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Graphique 3 : Violin plot 'Job' par 'Saving accounts' et 'Risk'
p3 <- ggplot(df, aes(x = `Saving accounts`, y = factor(Job), fill = Risk)) +
  geom_violin(position = position_dodge(width = 0.8), scale = "area") +
  scale_fill_brewer(palette = "Greens") +
  ggtitle("Violin Plot: Job by Saving accounts and Risk") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Affichage des 3 graphiques en colonne
grid.arrange(p1, p2, p3, ncol = 1, heights = c(1, 1, 1))



#SHOW BASIC STATS PER SAVING ACCOUNT

# Charger les bibliothèques nécessaires


# Appliquer summary à chaque groupe et chaque variable concernée
summary_by_saving <- df %>%
  group_by(`Saving accounts`) %>%
  summarise(across(c(Duration, Job, `Credit amount`), list(
    Min = ~min(., na.rm = TRUE),
    Q1 = ~quantile(., 0.25, na.rm = TRUE),
    Median = ~median(., na.rm = TRUE),
    Mean = ~mean(., na.rm = TRUE),
    Q3 = ~quantile(., 0.75, na.rm = TRUE),
    Max = ~max(., na.rm = TRUE),
    SD = ~sd(., na.rm = TRUE)
  ), .names = "{.col}_{.fn}"))

# Transposer le résultat comme en .T de pandas
summary_transposed <- summary_by_saving %>%
  pivot_longer(-`Saving accounts`, names_to = "Measure", values_to = "Value") %>%
  pivot_wider(names_from = `Saving accounts`, values_from = Value)

# Afficher le tableau
print(summary_transposed)


#ANALYSIS BY CREDIT CARD PURPOSE

# Packages nécessaires



# Count plot
p1 <- ggplot(df, aes(x = `Purpose`, fill = Risk)) +
  geom_bar(position = "dodge") +
  scale_fill_brewer(palette = "Pastel1") +  # Proche du "muted" de seaborn
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 10, hjust = 1)) +
  ggtitle("Count Plot: Purpose by Risk")

# Boxenplot (remplacer par geom_boxplot si vous n’avez pas ggplotBoxen)
# Si `ggplotBoxen` n’est pas disponible, utilisez simplement geom_boxplot()
# ou ggbump::geom_bump_box() si dispo
p2 <- ggplot(df, aes(x = `Purpose`, y = `Credit amount`, fill = Risk)) +
  geom_boxplot(position = position_dodge(width = 0.8)) +
  scale_fill_brewer(palette = "Pastel1") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 10, hjust = 1)) +
  ggtitle("Box Plot: Credit Amount by Purpose and Risk")

# Violin plot
p3 <- ggplot(df, aes(x = `Purpose`, y = Job, fill = Risk)) +
  geom_violin(position = position_dodge(width = 0.8), trim = FALSE) +
  scale_fill_brewer(palette = "Pastel1") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 10, hjust = 1)) +
  ggtitle("Violin Plot: Job by Purpose and Risk")

# Affichage des 3 graphiques verticalement
(p1 / p2 / p3) + plot_layout(heights = c(1, 1, 1)) & 
  plot_annotation(title = "Visualizations Grouped by Purpose (Hue = Risk)",
                  theme = theme(plot.title = element_text(hjust = 0.5, size = 16)))



#PER HOUSING


# Définir les couleurs manuellement : mauve et vin rose
custom_colors <- c("Bad" = "#8A2BE2",  # mauve (blueviolet)
                   "Good" = "#C71585") # vin rose (mediumvioletred)

# Count plot : Housing vs Risk
p1 <- ggplot(df, aes(x = Housing, fill = Risk)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = custom_colors) +
  ggtitle("Count Plot: Housing by Risk") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Boxenplot (approximé)
p2 <- ggplot(df, aes(x = Housing, y = `Credit amount`, fill = Risk)) +
  stat_halfeye(
    adjust = 0.5,
    width = 0.6,
    .width = 0,
    justification = -0.2,
    point_colour = NA,
    position = position_dodge(width = 0.8)
  ) +
  geom_boxplot(
    width = 0.2,
    outlier.shape = NA,
    position = position_dodge(width = 0.8)
  ) +
  scale_fill_manual(values = custom_colors) +
  ggtitle("Boxenplot: Credit amount by Housing and Risk") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Violin plot : Housing vs Job (Hue = Risk)
p3 <- ggplot(df, aes(x = Housing, y = Job, fill = Risk)) +
  geom_violin(position = position_dodge(width = 0.8)) +
  scale_fill_manual(values = custom_colors) +
  ggtitle("Violin Plot: Job by Housing and Risk") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Combinaison des 3 graphiques
(p1 / p2 / p3) + 
  plot_layout(heights = c(1, 1, 1)) + 
  plot_annotation(title = "Grouped Visualizations by Housing (Hue = Risk)",
                  theme = theme(plot.title = element_text(hjust = 0.5, size = 16)))

##CLUSTERING

#K-MEANS
#APPLYING ELBOW METHOD TO FIND THE BEST NUMBER OF CLUSTERS


# Supposons que num_df_scaled est votre dataframe/ matrice déjà normalisée

inertias <- numeric()

# Calculer la somme des carrés intra-clusters (tot.withinss) pour k de 2 à 15
for (i in 2:15) {
  set.seed(0)
  kmeans_result <- kmeans(df_scaled, centers = i, nstart = 25)
  inertias[i - 1] <- kmeans_result$tot.withinss
}

# Préparer un dataframe pour ggplot
df_elbow <- data.frame(
  k = 2:15,
  inertia = inertias
)

# Tracer le graphique du coude
ggplot(df_elbow, aes(x = k, y = inertia)) +
  geom_line(color = "steelblue", size = 1.2) +
  geom_point(color = "steelblue", size = 3) +
  ggtitle("ELBOW METHOD") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_x_continuous(breaks = 2:15)




#ALTERNATIVE METHOD: SILHOUTE SCORE WITH RANDOM SAMPLING


# Supposons que num_df_scaled est votre dataframe/matrice normalisée

results <- data.frame(num_cluster = integer(),
                      seed = integer(),
                      sil_score = numeric())

# Boucle pour k de 2 à 15 et seed de 0 à 19
for (i in 2:15) {
  for (r in 0:19) {
    set.seed(r)
    kmeans_result <- kmeans(num_df_scaled, centers = i, nstart = 25)
    c_labels <- kmeans_result$cluster
    sil <- silhouette(c_labels, dist(num_df_scaled))
    sil_ave <- mean(sil[, 3])
    results <- rbind(results, data.frame(num_cluster = i, seed = r, sil_score = sil_ave))
  }
}

# Pivot de données pour heatmap : lignes=num_cluster, colonnes=seed
pivot_kmeans <- reshape2::dcast(results, num_cluster ~ seed, value.var = "sil_score")
rownames(pivot_kmeans) <- pivot_kmeans$num_cluster
pivot_kmeans$num_cluster <- NULL

# Convertir en matrix pour heatmap
mat <- as.matrix(pivot_kmeans)


magma_palette <- viridis(100, option = "magma")

# Affichage heatmap avec pheatmap
pheatmap(mat,
         color = magma_palette,
         display_numbers = TRUE,
         number_format = "%.3f",
         fontsize_number = 8,
         cluster_rows = FALSE,
         cluster_cols = FALSE,
         main = "Silhouette Scores Heatmap (num_cluster vs seed)",
         border_color = "grey")

#The scores of 2,3,4 and 5 are pretty stable, Let's pick a number of cluster from that range.


#AT 3 NUMBER OF CLUSTERS


set.seed(0)
km <- kmeans(num_df_scaled, centers = 3, nstart = 25)
clusters <- km$cluster

# plot 3D avec rgl
plot3d(num_df_scaled[,1], num_df_scaled[,2], num_df_scaled[,3],
       col = clusters,
       size = 5,
       type = "s",
       xlab = colnames(num_df_scaled)[1],
       ylab = colnames(num_df_scaled)[2],
       zlab = colnames(num_df_scaled)[3],
       main = "3D Scatter Plot of KMeans Clusters")



# Assurez-vous que clusters est un facteur
df$clusters <- factor(clusters)

# Palette cividis (via viridis)
cividis_pal <- viridis::viridis(n = length(levels(df$clusters)), option = "C")

p1 <- ggplot(df, aes(x = Duration, y = `Credit amount`, color = clusters)) +
  geom_point() +
  scale_color_manual(values = cividis_pal) +
  theme_minimal() +
  ggtitle("Duration vs Credit amount")

p2 <- ggplot(df, aes(x = Age, y = `Credit amount`, color = clusters)) +
  geom_point() +
  scale_color_manual(values = cividis_pal) +
  theme_minimal() +
  ggtitle("Age vs Credit amount")

p3 <- ggplot(df, aes(x = Age, y = Duration, color = clusters)) +
  geom_point() +
  scale_color_manual(values = cividis_pal) +
  theme_minimal() +
  ggtitle("Age vs Duration")

grid.arrange(p1, p2, p3, ncol = 3)


#LET'S CREATE A DATAFRAME TO SUMMARIZE THE RESULT


# Créer un nouveau dataframe avec les colonnes d’intérêt et la variable cluster
df_clustered <- df %>%
  select(Age, Duration, `Credit amount`) %>%
  mutate(cluster = factor(clusters))

# Calculer la moyenne par cluster
df_clustered_summary <- df_clustered %>%
  group_by(cluster) %>%
  summarise(across(everything(), mean, na.rm = TRUE))

print(df_clustered_summary)
