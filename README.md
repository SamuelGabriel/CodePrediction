# R252 Project (sgm48)

## Summary
The projects core is to reimplement 'Learning Python Code Suggestion with a Sparse Pointer Network' by Bhoopchand et al. (https://arxiv.org/pdf/1611.08307.pdf)
## Overview of where code is
The main code is in `src`, which is based on the language model provided in class.
It incorporates the model described in my paper in `language_model`.
`container` is a container directory to build the container that I use with the script `coda.py` to run the training on codalabs GPUs.

## Project Plan
### LM with Attention on Trivially Prepared Dataset (Until Session 4)
> The target for session 4 is to have an implementation of a language model, that, different from the model implemented as our practical, uses an attention mechanism to predict the next token. The knowledge gained here can be used to better understand the implementation of the paper. This also is an interesting baseline that could be further explored.
> While the dataset used in the paper is preprocessed in a quite complex way, especially in differentiating between different kinds of identifiers based on where they are introduced and respecting the scope of identifiers for name replacement, my dataset preparation in this and the next week should not differentiate between the different kinds of identifiers, and just replace identifiers with tokens independent of what kind of identifier they are.
>
> -- Original Proposal

I took another path in this week and ended up with already having a reimplementation of 'Sparse Pointer Networks', but I could not see any improvements so far compared to LSTM.
Done at the end of this week:
- Make source code from the paper work with the pipeline provided in class: change python version and tf version
- Implemented a pipeline to mask non-identifiers for the copying mechanism
- Have a GPU pipeline with Azure and Codalab
Risks for the future:
- Unsuccessful bug hunt: no improvement in performance over LSTM

### Reimplementation of a 'Sparse Pointer Network' on a Trivially Prepared Dataset (Until Session 5)
> The goal for session 5 is to finish a reimplementation of the SPN model (Bhoopchand et al.), based on the knowledge gained from the previous week. 
>
> -- Original Proposal

Done this week:
- New Dataset split, with a 5x increased training set size
- Still no improvement even for larger set
- Found reason for bad training performance: the lambda parameter that is the gate between choosing to use the predictions of the standard LSTM and the attention copying seems to be trained towards only allowing LSTM predictions fairly quick. Thus after first epoch it is almost 0 for the attention copying for all examples in a sample I took.
    - This problem is the same for attention over inputs and outputs
- Tested if the copying mechanism does anything useful, by letting a model with only the copying run on a small dataset and could see that especially for input-based attention the copy mechanism can be useful. It achieves a 20% accuracy on the mini data set vs 70% of of the LSTM and 5% of a random model.
- Dataset preperation, the task for session 6. I have a prototype of a transforming function, but it is very slow. Tomorrow I will spend half of the day to try to improve efficiency so that it runs over the whole code as well as fix some last bugs.

### Proper Dataset Preparation (Until Session 6)
> In this week the plan is to come as close as possible to the way the dataset was prepared in the paper. There might be cases where this is hard, for a lot of identifiers in the dataset we use in class it is not totally clear which of the types of identifiers they belong to and I  might need to come up with some tricks to figure that out.\\
>
> -- Original Proposal

Original task already done in the previous work. Focus of this week now: Why does the SPN not perform better?
Tasks:
- Propagate hidden state between sequences – or simply try to use longer sequences
- Implement competing attention mechanism
- New dataset that also takes types into consideration

### Extensions and Experiments (Until Session 7)
> I want to propose two extension ideas, to be finished as much as possible until session 7.
> Firstly, it might be interesting to try sharing an embedding between all identifiers (of some kind). This might lead to interesting results since there already is random embedding sharing between identifiers in the SPN model. To have the same identifier and still predict usable tokens would be possible in the SPN, since the addressing of the pointers works over the output of the LSTM and not the embeddings directly. Thus I think it might lead to similar results to just replace all the embeddings with a single embedding, or at least replace all embeddings of identifiers of one type with a single embedding.
> Another possibility would be to test the effects on the architecture when removing the filter on the SPN making it similar to a simple combination of a Pointer Network and a standard LSTM (with attention) like proposed by See et al. for summarization. This simpler approach, which treats every token equally, was not introduced as a baseline in the original paper.
>
> -- Original Proposal

Working on:
- trying all the last resorts to make a difference: attention over outputs, larger attention window, look into attention storage

Next:
- different way of propagating error: learn to - only - copy whenever possible

### Padding (Until Session 8)
No proposal yet...
