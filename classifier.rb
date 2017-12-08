abstracts  = SmarterCSV.process('lg_abstracts.csv')
titles     = SmarterCSV.process('lg_titles.csv')
classifier = ClassifierReborn::Bayes.new 'Cancer', 'Not Cancer'

input.each do |i|
  if i[:abstract]
    if i[:is_cancer] == 'true'
      classifier.train 'Cancer', i[:abstract]
    else
      classifier.train 'Not Cancer', i[:abstract]
    end
  end
end

input2.each do |i|
  if i[:title]
    if i[:is_cancer] == 'true'
      classifier.train 'Cancer', i[:title]
    else
      classifier.train 'Not Cancer', i[:title]
    end
  end
end

binding.pry
