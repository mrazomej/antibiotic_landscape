function Div(el)
    -- Check if the div has a specific class, e.g., 'graybox'
    if el.classes:includes("graybox") then
        -- Convert the div to a custom LaTeX tcolorbox
        local latexStart = pandoc.RawBlock('latex', '\\begin{graybox}')
        local latexEnd = pandoc.RawBlock('latex', '\\end{graybox}')

        -- Insert the LaTeX start and end commands around the div content
        table.insert(el.content, 1, latexStart)
        table.insert(el.content, latexEnd)

        return el.content
    end
end
