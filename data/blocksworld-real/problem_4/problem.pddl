(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        green - block
        yellow - block
        blue - block
        purple - block
        red - block
        orange - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable green)
        (on yellow green)
        (on blue yellow)
        (on purple blue)
        (on red purple)
        (on orange red)
        (clear orange)
        ; green -> yellow -> blue -> purple -> red -> orange
    )
    (:goal (and (on purple green)
        (on orange yellow)
        (on red orange)
        (on blue red)))
)
    