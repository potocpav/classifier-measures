
extern crate classifier_measures;

use classifier_measures::roc_auc;


#[derive(PartialEq)]
enum Sex { M, F }

struct Person {
    name: &'static str,
    sex: Sex,
}

const FEMALES: [&str; 50] =
    ["Kyra", "Kaylie", "Angelique", "Mckenzie", "Genevieve", "Liliana", "Kamila", "Kylee", "Amani",
    "Liana", "Jaycee", "Athena", "Crystal", "Sage", "Francesca", "Lindsey", "Makena", "Amina",
    "Sofia", "Hana", "Giselle", "Marley", "Shelby", "Lucy", "Giuliana", "Liberty", "Nataly",
    "Marlie", "Leticia", "Belen", "Miley", "Ariella", "Evelin", "Saniya", "Erin", "Priscilla",
    "Elisa", "Destinee", "Nathalia", "Julianna", "Isabella", "Paloma", "Christina", "Jaylene",
    "Essence", "Zoe", "Cristina", "Shannon", "Aurora", "Alana"];

const MALES: [&str; 50] =
    ["Winston", "Issac", "James", "Braylen", "Reid", "Omar", "Dallas", "Anthony", "Jayce", "Bryce",
    "Dane", "Yusuf", "Desmond", "Tony", "Kade", "Elvis", "Ronnie", "Alessandro", "Kadyn", "Jaydan",
    "Alexander", "Brandon", "Paxton", "Shaun", "Ali", "Grayson", "Francis", "Jamie", "Luca",
    "Izaiah", "Keshawn", "Terrell", "Ricky", "Jermaine", "Donovan", "John", "Frankie", "Thaddeus",
    "Nathen", "King", "Aiden", "Joel", "Marcel", "Prince", "Aryan", "Jasiah", "Phoenix", "Mathew",
    "Isaac", "Kane"];

fn get_people() -> Vec<Person> {
    FEMALES.into_iter()
        .map(|name| Person { name, sex: Sex::F })
        .chain(MALES.into_iter()
        .map(|name| Person { name, sex: Sex::M }))
        .collect::<Vec<_>>()
}


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
