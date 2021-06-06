# HKT
Humor Knowledge Enriched Transformer



## Pretrained Models
To test the performances of the best models on UR-FUNNY and MUsTARD please follow the instructrtions. We have reported these best perormances in the paper. 

Test HKT model on UR-FUNNY dtaset:

python test_pretrained_models.py --dataset="humor" --max_seq_length=64 --cross_n_heads=2 --fusion_dim=300

Output: Accuracy: 77.36, F score:  77.36

Test HKT model on MUsTARD dtaset:

python test_pretrained_models.py --dataset="sarcasm" --max_seq_length=77 --cross_n_heads=2 --fusion_dim=172

Output: Accuracy:79.41, F Score : 79.25
  


