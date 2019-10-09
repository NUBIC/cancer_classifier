# {:Accuracy=>0.2127193721157409, :Precision=>1.0, :Recall=>0.8509634820105108, :F1=>0.9194816540477577}

require 'naivebayes'
require 'smarter_csv'
require 'words_counted'

def extract_feature_hash(string)
  counter = WordsCounted.count(string)
  counter.token_frequency.inject({}) { |feature, word_count|
    feature.merge({ word_count[0] => word_count[1] })
  }
end

abstracts  = SmarterCSV.process('lg_abstracts.csv')
titles     = SmarterCSV.process('lg_titles.csv')

classifier = NaiveBayes::Classifier.new(model: 'Complement')

texts = titles.each_with_index.map{ |title, index|
  title[:title] == nil || abstracts[index][:abstract] == nil ?
      nil :
      { text: title[:title] + ' ' + abstracts[index][:abstract], is_cancer: title[:is_cancer] }
}.compact
texts.shuffle!
quadrants = []
4.times{ |i|
  quadrants[i] = texts[i*(texts.count/4)..((i+1)*(texts.count/4)-1)]
}

3.times { |i|
  quadrants[i].each{ |texts|
    binding.pry if texts[:text] == nil
    classifier.train(texts[:is_cancer], extract_feature_hash(texts[:text])) }
}

confusion_matrix = { true_positive: 0, true_negative: 0, false_positive: 0, false_negative: 0 }

quadrants[3].each { |texts|
  result = classifier.classify(extract_feature_hash(texts[:text])).max[0]

  if result == texts[:is_cancer]
    if texts[:is_cancer] == 'true'
      confusion_matrix[:true_positive] += 1
    else
      confusion_matrix[:true_negative] += 1
    end
  else
    if texts[:is_cancer] == 'true'
      confusion_matrix[:false_positive] += 1
    else
      confusion_matrix[:false_negative] += 1
    end
  end
}

metrics = {
    Accuracy: (confusion_matrix[:true_positive].to_f + confusion_matrix[:true_negative]) / texts.count,
    Precision: confusion_matrix[:true_positive].to_f / (confusion_matrix[:true_positive] + confusion_matrix[:false_positive]),
    Recall: confusion_matrix[:true_positive].to_f / (confusion_matrix[:true_positive] + confusion_matrix[:false_negative])
}
metrics.merge({ F1: (2 * metrics[:Precision] * metrics[:Recall]) / (metrics[:Precision] + metrics[:Recall]) })

binding.pry
