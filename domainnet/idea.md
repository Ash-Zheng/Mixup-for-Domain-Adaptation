## Mixup DA

> mixup的融合系数:2 - 2 / (1 + exp(- gamma * p)),从1->0, gamma = 10, p = steps / total_steps  
> algorithm:
> input: batch of labeled source samples and their label x = (x_b, y_b), batch of unlabeled  
> target samples u = (x_t), sharpening temperature T, number of augmentations K, mix parameter  
> lambda
> for b = 1 to B do  
>   for k = 1 to K do  
>       x_b_k_hat = Augment(x_b)  
>       x_t_k_hat = Augment(x_t)  
>   end for  
>   q_t_hat = 1 / K(P_model(y|x_t_k_hat; theta))  
>   q_t = Sharpen(q_t_hat, T)  
> end for  
> X = (x_b_k_hat, y_b)  
> U = (x_t_k_hat, q_b)  
> X_prime = (mixup(X_i, U_i), i \in (1, ..., |X|))  
> return X_prime  

> mixup(x1, x2):  
> x = lambda * x1 + (1 - lambda) * x2   
> y = lambda * y1 + (1 - lambda) * y2  
> return x, y

