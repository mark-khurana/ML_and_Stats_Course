local notes = {}

local function parse_bib_notes(bibpath)
  local f = io.open(bibpath, "r")
  if not f then
    f = io.open(bibpath:match("[^/]+$"), "r")
  end
  if not f then return end
  local content = f:read("*a")
  f:close()
  for entry in content:gmatch("@%w+%b{}") do
    local key = entry:match("@%w+{([%w_]+),")
    if key then
      local note = entry:match("note%s*=%s*{(.-)}")
      if note then
        notes[key] = note
      end
    end
  end
end

local function append_notes(blocks)
  return blocks:walk {
    Div = function (el)
      if el.classes:includes("csl-entry") and el.identifier:match("^ref%-") then
        -- The section-bibliographies filter suffixes reference identifiers
        -- with `--<section number>` to keep them unique across per-section
        -- bibliographies, so strip that before looking the key up.
        local key = el.identifier:gsub("^ref%-", ""):gsub("%-%-.*$", "")
        if notes[key] then
          local last = el.content[#el.content]
          if last and last.t == "Para" then
            last.content:insert(pandoc.Space())
            last.content:insert(pandoc.Span(
              {pandoc.Emph({pandoc.Str(notes[key])})},
              pandoc.Attr("", {"bib-note"})
            ))
            -- Reassign: indexing a Div's content yields a copy, so mutating
            -- `last` in place would otherwise be discarded.
            el.content[#el.content] = last
            return el
          end
        end
      end
    end
  }
end

--- True if a bibliography has already been rendered into the document,
--- i.e. some earlier filter has already run citeproc.
local function bibliography_already_rendered(doc)
  local found = false
  doc:walk {
    Div = function (el)
      if el.classes:includes("csl-entry") then
        found = true
      end
    end
  }
  return found
end

return {{
  Pandoc = function (doc)
    local bib = doc.meta.bibliography
    if bib then
      local path
      if pandoc.utils.type(bib) == "List" then
        path = pandoc.utils.stringify(bib[1])
      else
        path = pandoc.utils.stringify(bib)
      end
      parse_bib_notes(path)
    end

    -- Only run citeproc if nothing has resolved the citations yet.
    --
    -- The section-bibliographies filter runs before this one and, for any
    -- document whose body contains a level-1 section, resolves citations
    -- itself after suffixing every key with `--<section number>`. Running
    -- citeproc again here would try to re-resolve those suffixed keys against
    -- the unsuffixed bibliography, and every citation in the chapter would
    -- render as a broken `key--1?`. Guarding the call keeps both paths safe.
    if not bibliography_already_rendered(doc) then
      doc = pandoc.utils.citeproc(doc)
    end

    doc.blocks = append_notes(doc.blocks)
    return doc
  end
}}
