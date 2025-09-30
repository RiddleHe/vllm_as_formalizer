(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        purple - block
        blue - block
        green - block
        yellow - block
        red - block
        orange - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable purple)
        (on blue purple)
        (on green blue)
        (on yellow green)
        (on red yellow)
        (on orange red)
        (clear orange)
        ; purple -> blue -> green -> yellow -> red -> orange
    )
    (:goal (and (on green blue)
        (on purple orange)
        (on red yellow)))
)
    