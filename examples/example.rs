
extern crate classifier_measures;

mod data;

use data::{Sex, get_people};
use classifier_measures::roc_auc;

fn main() {
    let data = get_people();
    println!("Gender classifier strengths (ROC AUC):");

    println!("  Name length: {:.4}",
        roc_auc(&data, |p| (p.sex == Sex::F, p.name.len() as f64)).unwrap());

    println!("  Number of vowels in a name: {:.4}",
        roc_auc(&data, |p| (
            p.sex == Sex::F,
            p.name.bytes().filter(|&c| "aeiouy".bytes().any(|d| d == c)).count() as f64)
        ).unwrap());

    println!("  Name ends with a vowel: {:.4}",
        roc_auc(&data, |p| (
            p.sex == Sex::F,
            "aeiouy".bytes().filter(|&c| Some(c) == p.name.bytes().rev().next()).count() as f64
        )).unwrap());
}
