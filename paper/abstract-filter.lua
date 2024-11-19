function Div(el)
    -- Check if the div has the 'abstract' class
    if el.classes:includes("abstract") then
        -- Apply modifications if the output format is PDF
        if FORMAT:match("latex") then
            -- Define the LaTeX abstract environment
            local abstractStart = pandoc.RawBlock('latex', '\\begin{abstract}')
            local abstractEnd = pandoc.RawBlock('latex', '\\end{abstract}')
            -- Insert the LaTeX commands around the abstract content
            table.insert(el.content, 1, abstractStart)
            table.insert(el.content, abstractEnd)
            return el.content
        elseif FORMAT:match("html") then
            return el.content
        end
    end
    -- For other divs or formats, return unchanged
    return el
end
