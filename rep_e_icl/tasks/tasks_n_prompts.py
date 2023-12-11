DATASET_NAMES = [ 
"ade_corpus_v2-classification",
"ethos-directed_vs_generalized", 
"ethos-disability", 
"ethos-gender",
"ethos-national_origin", 
"ethos-race",
"ethos-religion", 
"ethos-sexual_orientation", 
"financial_phrasebank",
"glue-mrpc" ,
"glue-rte", 
"glue-wnli",
"hate_speech18",
"medical_questions_pairs",
"poem_sentiment" ,
"rotten_tomatoes", 
"sick",
"superglue-cb", 
"tweet_eval-hate",  
"tweet_eval-stance_atheism",
"tweet_eval-stance_feminist",
"tweet_eval-offensive"
]


TASKS_TO_PROMPTS = { 
                   
"ade_corpus_v2-classification": "Answer whether the drug and the adverse effect are related. ",
"rotten_tomatoes" : "Predict the sentiment of the following movie review. "
                   
}