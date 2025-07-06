# Modelli di Machine Learning per la Diagnosi del Diabete: Uno Studio Comparativo

Identificare precocemente i soggetti a rischio di diabete attraverso l'analisi di dati clinici e anagrafici è fondamentale per l'assistenza sanitaria preventiva. Nel corso dello studio, è stata eseguita un'analisi esplorativa del dataset clinico e sono state applicate tecniche di pre-processing per la gestione dei valori mancanti, l'eliminazione delle anomalie, il bilanciamento delle classi, la standardizzazione e la riduzione della dimensionalità. Sono stati quindi confrontati diversi modelli predittivi per l'identificazione del diabete: K-Nearest Neighbors, Support Vector Machine, Multi-Layer Perceptron, Random Forest e Histogram Gradient Boosting. Le performance, valutate tramite F1-score, hanno mostrato le migliori prestazioni per gli ultimi 2 modelli. I risultati ottenuti in fase di test confermano il potenziale delle tecniche di Machine Learning nell'ambito della diagnosi precoce del diabete. Sono tuttavia da tenere in considerazione le norme relative alla privacy e al trattamento di dati sensibili.

## Funzionalità

- Analisi dataset
- Pre-processing
- Training e predizione modelli ML

## Struttura

- **Data/**   
  Cartella contenente dataset in `.csv`, analisi esplorativa del training set (con grafici) e file di pre-processing (con visualizzazioni e funzione).
- **Modelli/**   
  Cartella contenente gli eseguibili dei modelli per addestramento e previsione, con risultati.

## Installazione

```bash
# Clona il repository
git clone  https://github.com/MaddalenaGhiotti/BusinessIntelligenceProject.git


# Installa le dipendenze Python
pip install -r requirements.txt
```

## Authors

- Stefano Caprioli (s339841)
- Martina Cristiani (s348736)
- Fabio Daniele Diena (s332743)
- Maddalena Ghiotti (s332834)
- Alberto Prino (s348174)


