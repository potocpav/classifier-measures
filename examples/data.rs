
#[derive(PartialEq)]
pub enum Sex { M, F }

pub struct Person {
    pub name: &'static str,
    pub sex: Sex,
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

pub fn get_people() -> Vec<Person> {
    FEMALES.into_iter()
        .map(|name| Person { name, sex: Sex::F })
        .chain(MALES.into_iter()
        .map(|name| Person { name, sex: Sex::M }))
        .collect::<Vec<_>>()
}
