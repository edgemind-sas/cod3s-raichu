//! Identifier grammar shared by every name a document declares.
//!
//! Instance, type, fragment, mode, feared-event and library names may be
//! any non-empty string (spaces and accents included: the delivered STRA
//! model names its instances `Algo Feu` and `Véhicule`) except that they
//! may not contain a `.`, a `$` or a control character. The `.` is what
//! keeps a public name `Instance.mode` unambiguous, and the `$` is
//! reserved for the templating grammar (`$self`, `$param`).

/// Checks a declared name against the grammar; the error explains the
/// rule, the caller adds where the name was found.
pub fn check_name(what: &str, name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err(format!("{what} name is empty"));
    }
    if let Some(bad) = name
        .chars()
        .find(|c| *c == '.' || *c == '$' || c.is_control())
    {
        let shown = if bad.is_control() {
            format!("control character U+{:04X}", u32::from(bad))
        } else {
            format!("`{bad}`")
        };
        return Err(format!(
            "{what} name `{name}` contains {shown}; a name may contain any character except `.`, `$` and control characters"
        ));
    }
    Ok(())
}

/// Splits a public name `Instance.mode` (or `Instance.port`) into its two
/// parts; `None` unless there is exactly one `.` with a non-empty part on
/// each side.
#[must_use]
pub fn split_public(name: &str) -> Option<(&str, &str)> {
    let (head, tail) = name.split_once('.')?;
    if head.is_empty() || tail.is_empty() || tail.contains('.') {
        return None;
    }
    Some((head, tail))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spaces_and_accents_are_names() {
        assert!(check_name("instance", "Algo Feu").is_ok());
        assert!(check_name("instance", "Véhicule").is_ok());
    }

    #[test]
    fn dot_dollar_control_and_empty_are_refused() {
        assert!(check_name("instance", "Srv.main").is_err());
        assert!(check_name("instance", "$x").is_err());
        assert!(check_name("instance", "a\nb").is_err());
        assert!(check_name("instance", "").is_err());
    }

    #[test]
    fn public_names_split_once() {
        assert_eq!(split_public("Srv.mdc_a"), Some(("Srv", "mdc_a")));
        assert_eq!(
            split_public("Algo Feu.mc_intru_algo"),
            Some(("Algo Feu", "mc_intru_algo"))
        );
        assert_eq!(split_public("Srv"), None);
        assert_eq!(split_public("a.b.c"), None);
        assert_eq!(split_public(".b"), None);
    }
}
