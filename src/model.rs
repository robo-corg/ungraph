use crate::summary::Summary;

pub trait Model {
    fn summary<'a>(&'a self, filename: Option<&'a str>) -> Box<dyn Summary + 'a>;
}
