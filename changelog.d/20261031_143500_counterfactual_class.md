### Changed

- The counterfactual facts that the local explanation prompts read name the class that the prediction changes to (`new_predicted_class`), and the likelihood of that class is `likelihood_of_new_predicted_class` (`probability_of_new_predicted_class` for a data scientist). The LLM no longer has to guess which class the likelihood is about.
