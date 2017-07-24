
#![cfg(test)]

use super::*;

#[test]
fn roc_auc_test_corner_cases() {
    // The same results as python's sklearn.metrics.roc_auc_score
    let tests = vec![
        (vec![0.0, 1.0, 0.0], vec![1.0, 2.0, 2.0], 0.75),
        (vec![0.0, 0.0, 1.0], vec![1.0, 2.0, 2.0], 0.75),
        (vec![0.0, 0.0, 1.0, 1.0], vec![1.0, 2.0, 2.0, 3.0], 0.875),
        (vec![0.0, 1.0, 0.0, 1.0], vec![1.0, 2.0, 2.0, 3.0], 0.875),
        (vec![0.0, 0.0, 1.0, 0.0], vec![2.0, 1.0, 1.0, 0.0], 0.5),
        (vec![0.0, 0.0, 1.0], vec![2.0, 1.0, 1.0], 0.25),
    ];
    for test in tests {
        println!("test: {:?}", test);
        let mut tests_zipped: Vec<_> = test.0.iter().cloned().map(|a| a == 1.0).zip(test.1.iter().cloned()).collect();
        assert_eq!(roc_auc_mut(&mut tests_zipped), Some(test.2));
    }
}


#[test]
fn pr_auc_test_corner_cases() {
    // The same results as python's sklearn.metrics.average_precision_score
    let tests = vec![
        (vec![0.0, 1.0, 0.0], vec![1.0, 2.0, 2.0], 0.75),
        (vec![0.0, 0.0, 1.0], vec![1.0, 2.0, 2.0], 0.75),
        (vec![0.0, 0.0, 1.0, 1.0], vec![1.0, 2.0, 2.0, 3.0], 11.0/12.0),
        (vec![0.0, 1.0, 0.0, 1.0], vec![1.0, 2.0, 2.0, 3.0], 11.0/12.0),
        (vec![0.0, 0.0, 1.0, 0.0], vec![2.0, 1.0, 1.0, 0.0], 1.0/6.0),
        (vec![0.0, 0.0, 1.0], vec![2.0, 1.0, 1.0], 1.0/6.0),
    ];
    for test in tests {
        println!("test: {:?}", test);
        let mut tests_zipped: Vec<_> = test.0.iter().cloned().map(|a| a == 1.0).zip(test.1.iter().cloned()).collect();
        assert_eq!(pr_auc_mut(&mut tests_zipped), Some(test.2));
    }
}
