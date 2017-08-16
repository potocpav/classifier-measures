
#![cfg(test)]

use super::*;

#[test]
fn roc_auc_test_corner_cases() {
    // The same results as python's sklearn.metrics.roc_auc_score
    let tests = vec![
        (vec![], vec![], None),
        (vec![false], vec![1.0], None),
        (vec![true, true], vec![1.0, 2.0], None),
        (vec![true, false], vec![1.0, ::std::f64::NAN], None),
        (vec![true, false], vec![1.0, ::std::f64::INFINITY], None),

        (vec![false, true, false], vec![1.0, 2.0, 2.0], Some(0.75)),
        (vec![false, false, true], vec![1.0, 2.0, 2.0], Some(0.75)),
        (vec![false, false, true, true], vec![1.0, 2.0, 2.0, 3.0], Some(0.875)),
        (vec![false, true, false, true], vec![1.0, 2.0, 2.0, 3.0], Some(0.875)),
        (vec![false, false, true, false], vec![2.0, 1.0, 1.0, 0.0], Some(0.5)),
        (vec![false, false, true], vec![2.0, 1.0, 1.0], Some(0.25)),
    ];
    for test in tests {
        println!("test: {:?}", test);
        let mut tests_zipped: Vec<_> = test.0.iter().cloned().zip(test.1.iter().cloned()).collect();
        assert_eq!(roc_auc(&tests_zipped, |&x| x), test.2);
        assert_eq!(roc_auc_mut(&mut tests_zipped), test.2);
    }
}


#[test]
fn pr_auc_test_corner_cases() {
    // The same results as python's sklearn.metrics.average_precision_score
    let tests = vec![
        (vec![], vec![], None),
        (vec![false], vec![1.0], None),
        (vec![true, true], vec![1.0, 2.0], None),
        (vec![true, false], vec![1.0, ::std::f64::NAN], None),
        (vec![true, false], vec![1.0, ::std::f64::INFINITY], None),

        (vec![false, true, false], vec![1.0, 2.0, 2.0], Some(0.75)),
        (vec![false, false, true], vec![1.0, 2.0, 2.0], Some(0.75)),
        (vec![false, false, true, true], vec![1.0, 2.0, 2.0, 3.0], Some(11.0/12.0)),
        (vec![false, true, false, true], vec![1.0, 2.0, 2.0, 3.0], Some(11.0/12.0)),
        (vec![false, false, true, false], vec![2.0, 1.0, 1.0, 0.0], Some(1.0/6.0)),
        (vec![false, false, true], vec![2.0, 1.0, 1.0], Some(1.0/6.0)),
    ];
    for test in tests {
        println!("test: {:?}", test);
        let mut tests_zipped: Vec<_> = test.0.iter().cloned().zip(test.1.iter().cloned()).collect();
        assert_eq!(pr_auc(&tests_zipped, |&x| x), test.2);
        assert_eq!(pr_auc_mut(&mut tests_zipped), test.2);
    }
}
