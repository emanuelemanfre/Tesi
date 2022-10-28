# Tesi
Per replicare i risultati della tesi bisogna:
1) Lanciare la CyclaGAN (il main è nel train.py) dando come immagini di train tutte le visibili (o infrarossi a seconda di ciò che si vuole generare) senza labellarle. Le immagini fake verranno salvate su un folder il cui nome è indicato nel file train.py, bisogna quindi prima creare questo folder.
2) Nel folder "SiameseNetwork" ci sono i due modelli allenati su immagini visibili (siamese_VIS) e termiche (siamese_IR), i sei (per tipo di rete) modelli salvati, i dataset divisi in visibile e termico e labellati per classe. Un dataset di sole immagini visibili fake uno di ir fake. 
3) Nel file metriche ci sono tutti gli script per le metriche di valutazioni delle immagini generate. 
